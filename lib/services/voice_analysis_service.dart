import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fftea/fftea.dart';
import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
import 'package:flutter/services.dart' show rootBundle;
import 'package:record/record.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

const int kNFft = 2048;
const int kHopLength = 512;
const int kNMels = 128;
const int kSampleRate = 16000;
const int kNFreqBins = kNFft ~/ 2 + 1; // 1025

// Number of consecutive 500ms windows that must agree on an *escalated*
// tier (Elevated or HighDanger) before that tier is allowed to affect the
// displayed classification or the score formula's branch. This does not
// change the alpha=0.3 smoothing constant from Chapter 4.5.1's pseudocode
// - it only gates which formula branch/classification is used each cycle,
// so a single loud syllable can't instantly flip the label or spike the
// score. Dropping back down to a calmer tier is always immediate.
const int kEscalationConfirmWindows = 2;

/// Which bundled model file the service loads. Retraining produces a new
/// file (see tools/voice_model/train_voice_model.py); switching to it is
/// a one-line change here plus listing it under `assets:` in pubspec.yaml.
const String kVoiceModelAsset = 'assets/ml/neuralsafe_voice_model_v3.tflite';

/// A 1-second window whose RMS amplitude (samples scaled to [-1, 1]) is
/// below this is treated as silence: the model is skipped and the window
/// is scored as pure Normal. This matters because the log-Mel features
/// are peak-normalized PER WINDOW (see MelSpectrogramExtractor.extract),
/// so a near-silent window is stretched to the same dynamic range as a
/// loud one and the network cannot tell them apart by volume. ~0.005 is
/// roughly -46 dBFS, below quiet-room speech on a phone mic. The value
/// is a starting point — the diagnostic log line printed from
/// _processWindow shows real RMS values so it can be tuned from data.
const double kSilenceRmsThreshold = 0.005;

/// In debug builds, print RMS + class probabilities every Nth window
/// (windows are 500 ms apart, so 4 means one line every 2 seconds).
const int kDiagnosticLogEveryNWindows = 4;

List<double> _buildHannWindow(int n) {
  return List<double>.generate(
    n,
    (i) => 0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)),
  );
}

class MelSpectrogramExtractor {
  MelSpectrogramExtractor(this.melFilterbank)
      : _fft = FFT(kNFft),
        _hann = _buildHannWindow(kNFft);

  final Float32List melFilterbank;
  final FFT _fft;
  final List<double> _hann;

  Float32List extract(List<double> audio) {
    assert(audio.length == kSampleRate);

    final padAmount = kNFft ~/ 2;
    final padded = List<double>.filled(audio.length + 2 * padAmount, 0.0);
    for (var i = 0; i < audio.length; i++) {
      padded[padAmount + i] = audio[i];
    }

    final nFrames = 1 + (padded.length - kNFft) ~/ kHopLength;
    final melPower =
        List.generate(kNMels, (_) => List<double>.filled(nFrames, 0.0));
    final frameBuffer = List<double>.filled(kNFft, 0.0);

    for (var f = 0; f < nFrames; f++) {
      final start = f * kHopLength;
      for (var i = 0; i < kNFft; i++) {
        frameBuffer[i] = padded[start + i] * _hann[i];
      }

      final freq = _fft.realFft(frameBuffer);
      final magnitudes = freq.discardConjugates().magnitudes();

      for (var mel = 0; mel < kNMels; mel++) {
        var sum = 0.0;
        final rowOffset = mel * kNFreqBins;
        for (var k = 0; k < kNFreqBins; k++) {
          final power = magnitudes[k] * magnitudes[k];
          sum += melFilterbank[rowOffset + k] * power;
        }
        melPower[mel][f] = sum;
      }
    }

    final logMel =
        List.generate(kNMels, (_) => List<double>.filled(nFrames, 0.0));
    var maxVal = double.negativeInfinity;
    for (var mel = 0; mel < kNMels; mel++) {
      for (var f = 0; f < nFrames; f++) {
        final v = 10 * math.log(math.max(melPower[mel][f], 1e-10)) / math.ln10;
        logMel[mel][f] = v;
        if (v > maxVal) maxVal = v;
      }
    }

    final result = Float32List(kNMels * nFrames);
    for (var mel = 0; mel < kNMels; mel++) {
      for (var f = 0; f < nFrames; f++) {
        var v = logMel[mel][f] - maxVal;
        if (v < -80.0) v = -80.0;
        result[mel * nFrames + f] = v;
      }
    }

    return result;
  }
}

class VoiceScoringState {
  double previousVoiceScore;
  int consecutiveElevatedCount;

  /// The tier (0=Normal, 1=Elevated, 2=HighDanger) currently "unlocked"
  /// for use in classification/score-branch selection, after passing the
  /// escalation confirmation gate. Starts at Normal.
  int tierConfirmed;

  /// Bookkeeping for the confirmation gate: the tier we're waiting to
  /// confirm, and how many consecutive windows have agreed on it.
  int tierPending;
  int tierPendingStreak;

  VoiceScoringState({
    this.previousVoiceScore = 0.0,
    this.consecutiveElevatedCount = 0,
    this.tierConfirmed = 0,
    this.tierPending = 0,
    this.tierPendingStreak = 0,
  });
}

class VoiceAnalysisResult {
  final double score;
  final String classification;
  final List<double> probabilities;

  VoiceAnalysisResult({
    required this.score,
    required this.classification,
    required this.probabilities,
  });

  @override
  String toString() =>
      'VoiceAnalysisResult(score: $score, class: $classification, probs: $probabilities)';
}

/// Instantaneous tier from this window's raw probabilities alone, using
/// the exact thresholds from Chapter 4.5.1's pseudocode. This is only
/// used as the *candidate* tier for the confirmation gate below - it is
/// not applied directly to the classification or score branch.
int _instantaneousTier(double pNormal, double pElevated, double pHighDanger) {
  if (pHighDanger > 0.6) return 2;
  if (pElevated > 0.5 && pElevated > pNormal) return 1;
  return 0;
}

VoiceAnalysisResult computeVoiceScore(
  List<double> classProbs,
  VoiceScoringState state,
) {
  assert(classProbs.length == 3);
  final pNormal = classProbs[0];
  final pElevated = classProbs[1];
  final pHighDanger = classProbs[2];

  final tentativeTier = _instantaneousTier(pNormal, pElevated, pHighDanger);

  if (tentativeTier <= state.tierConfirmed) {
    // Same tier, or a de-escalation: apply immediately (the safe direction
    // never needs to wait for confirmation), and reset the pending streak.
    state.tierConfirmed = tentativeTier;
    state.tierPendingStreak = 0;
  } else {
    // Escalation attempt: only let it through after it repeats for
    // kEscalationConfirmWindows consecutive windows.
    if (state.tierPending == tentativeTier) {
      state.tierPendingStreak++;
    } else {
      state.tierPending = tentativeTier;
      state.tierPendingStreak = 1;
    }
    if (state.tierPendingStreak >= kEscalationConfirmWindows) {
      state.tierConfirmed = tentativeTier;
    }
    // else: hold at the current confirmed tier for this cycle.
  }

  String classification;
  switch (state.tierConfirmed) {
    case 2:
      classification = 'HighDanger';
      break;
    case 1:
      classification = 'Elevated';
      break;
    default:
      classification = 'Normal';
  }

  double voiceScoreRaw;
  if (state.tierConfirmed == 2) {
    if (pHighDanger > 0.7) {
      voiceScoreRaw = 85 + (pHighDanger - 0.7) * 150;
    } else {
      voiceScoreRaw = 50 + math.max(0, pHighDanger - 0.5) * 70;
    }
  } else if (state.tierConfirmed == 1) {
    voiceScoreRaw = 30 + math.max(0, pElevated - 0.5) * 40;
  } else {
    voiceScoreRaw = pHighDanger * 15;
  }

  const alpha = 0.3;
  var voiceScoreSmoothed =
      alpha * voiceScoreRaw + (1 - alpha) * state.previousVoiceScore;

  if (voiceScoreSmoothed > 40) {
    state.consecutiveElevatedCount++;
    if (state.consecutiveElevatedCount >= 6) {
      voiceScoreSmoothed += 15;
    }
  } else {
    state.consecutiveElevatedCount = 0;
  }

  state.previousVoiceScore = voiceScoreSmoothed;

  final finalScore = voiceScoreSmoothed > 100 ? 100.0 : voiceScoreSmoothed;

  return VoiceAnalysisResult(
    score: finalScore,
    classification: classification,
    probabilities: classProbs,
  );
}

class VoiceAnalysisService {
  VoiceAnalysisService._();
  static final VoiceAnalysisService instance = VoiceAnalysisService._();

  MelSpectrogramExtractor? _extractor;
  Interpreter? _interpreter;
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _micSubscription;
  final List<int> _int16Buffer = [];
  int _windowCounter = 0;
  Timer? _windowTimer;
  final VoiceScoringState _state = VoiceScoringState();
  final StreamController<VoiceAnalysisResult> _scoreController =
      StreamController<VoiceAnalysisResult>.broadcast();

  bool _initialized = false;
  bool get isInitialized => _initialized;
  bool get isRunning => _windowTimer != null;

  Stream<VoiceAnalysisResult> get scores => _scoreController.stream;
  VoiceAnalysisResult? latest;

  Future<void> init() async {
    if (_initialized) return;

    final filterbankData =
        await rootBundle.load('assets/ml/mel_filterbank.bin');
    final melFilterbank = filterbankData.buffer.asFloat32List(
        filterbankData.offsetInBytes, filterbankData.lengthInBytes ~/ 4);
    _extractor = MelSpectrogramExtractor(melFilterbank);

    _interpreter = await Interpreter.fromAsset(kVoiceModelAsset);

    _initialized = true;
  }

  Future<bool> start() async {
    if (!_initialized) await init();
    if (isRunning) return true;

    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) return false;

    final stream = await _recorder.startStream(const RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: kSampleRate,
      numChannels: 1,
    ));

    _micSubscription = stream.listen(_onAudioChunk);
    _windowTimer = Timer.periodic(
        const Duration(milliseconds: 500), (_) => _processWindow());

    return true;
  }

  void _onAudioChunk(Uint8List chunk) {
    final byteData = ByteData.sublistView(chunk);
    for (var i = 0; i + 1 < chunk.length; i += 2) {
      _int16Buffer.add(byteData.getInt16(i, Endian.little));
    }

    final maxSamples = kSampleRate * 2;
    if (_int16Buffer.length > maxSamples) {
      _int16Buffer.removeRange(0, _int16Buffer.length - maxSamples);
    }
  }

  void _processWindow() {
    if (_int16Buffer.length < kSampleRate) return;

    final windowSamples =
        _int16Buffer.sublist(_int16Buffer.length - kSampleRate);
    final floatSamples =
        windowSamples.map((s) => s / 32768.0).toList(growable: false);

    _windowCounter++;
    var sumSquares = 0.0;
    for (final s in floatSamples) {
      sumSquares += s * s;
    }
    final rms = math.sqrt(sumSquares / floatSamples.length);
    final shouldLog =
        kDebugMode && _windowCounter % kDiagnosticLogEveryNWindows == 0;

    if (rms < kSilenceRmsThreshold) {
      // Silence gate — see kSilenceRmsThreshold. Still goes through
      // computeVoiceScore so smoothing/escalation state decays normally.
      final silentResult =
          computeVoiceScore(const [1.0, 0.0, 0.0], _state);
      if (shouldLog) {
        debugPrint('[NeuralSafe][voice] rms=${rms.toStringAsFixed(4)} '
            '(silence gate) score=${silentResult.score.toStringAsFixed(1)}');
      }
      latest = silentResult;
      if (!_scoreController.isClosed) {
        _scoreController.add(silentResult);
      }
      return;
    }

    final rawLogMel = _extractor!.extract(floatSamples);
    final normalized = Float32List(rawLogMel.length);
    for (var i = 0; i < rawLogMel.length; i++) {
      normalized[i] = (rawLogMel[i] + 80.0) / 80.0;
    }

    final nFrames = rawLogMel.length ~/ kNMels;
    final input = [
      List.generate(
        kNMels,
        (mel) => List.generate(
          nFrames,
          (t) => [normalized[mel * nFrames + t]],
        ),
      )
    ];
    final output = List.filled(1 * 3, 0.0).reshape([1, 3]);
    _interpreter!.run(input, output);
    final probs = (output[0] as List).map((e) => e as double).toList();

    final result = computeVoiceScore(probs, _state);
    if (shouldLog) {
      debugPrint('[NeuralSafe][voice] rms=${rms.toStringAsFixed(4)} '
          'probs[N,E,H]=${probs.map((p) => p.toStringAsFixed(2)).toList()} '
          'class=${result.classification} score=${result.score.toStringAsFixed(1)}');
    }
    latest = result;
    if (!_scoreController.isClosed) {
      _scoreController.add(result);
    }
  }

  Future<void> stop() async {
    _windowTimer?.cancel();
    _windowTimer = null;
    await _micSubscription?.cancel();
    _micSubscription = null;
    if (await _recorder.isRecording()) {
      await _recorder.stop();
    }
    _int16Buffer.clear();
  }

  Future<void> dispose() async {
    await stop();
    await _scoreController.close();
    _recorder.dispose();
  }
}
