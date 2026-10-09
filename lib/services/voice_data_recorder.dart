import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'voice_analysis_service.dart';

/// Training-data classes for the voice model. These map 1:1 to the
/// model's three output classes (index 0/1/2 = Normal/Elevated/HighDanger)
/// — see tools/voice_model/train_voice_model.py.
enum VoiceClipLabel { normal, elevated, danger }

/// Records labeled audio clips for retraining the voice model, using the
/// SAME capture path the live model sees: `AudioRecorder.startStream`
/// with 16 kHz mono 16-bit PCM from the default microphone source. That
/// matters — a model trained on clips from a different recorder app
/// (different gain, noise suppression, codec) would be learning from a
/// different microphone than the one it runs on.
///
/// Clips are written as standard WAV files into
/// `<external app dir>/voice_data/<label>/`, which `adb pull` can read
/// without root (path is exposed via [outputDirectory]).
///
/// This is a development/data-collection tool, not part of the safety
/// pipeline. It refuses to run while Active Safety Monitoring has the
/// microphone open.
class VoiceDataRecorder {
  VoiceDataRecorder._();
  static final VoiceDataRecorder instance = VoiceDataRecorder._();

  static const int _sampleRate = 16000;

  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;
  bool get isRecording => _isRecording;

  Future<Directory> outputDirectory() async {
    final external = await getExternalStorageDirectory();
    final root = external ?? await getApplicationDocumentsDirectory();
    final dir = Directory('${root.path}/voice_data');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  /// Number of saved clips per label.
  Future<Map<VoiceClipLabel, int>> clipCounts() async {
    final base = await outputDirectory();
    final counts = <VoiceClipLabel, int>{};
    for (final label in VoiceClipLabel.values) {
      final dir = Directory('${base.path}/${label.name}');
      counts[label] = await dir.exists()
          ? (await dir.list().where((e) => e.path.endsWith('.wav')).length)
          : 0;
    }
    return counts;
  }

  /// Records one clip. [onStatus] receives short human-readable progress
  /// messages ("Get ready... 3", "Recording... 7s left"). Returns the
  /// saved file path. Throws [StateError] with a user-readable message if
  /// recording can't start.
  Future<String> recordClip({
    required VoiceClipLabel label,
    Duration countdown = const Duration(seconds: 3),
    Duration duration = const Duration(seconds: 10),
    void Function(String status)? onStatus,
  }) async {
    if (_isRecording) {
      throw StateError('Already recording a clip.');
    }
    if (VoiceAnalysisService.instance.isRunning) {
      throw StateError(
          'Turn OFF Active Safety Monitoring first — it is using the microphone.');
    }
    if (!await _recorder.hasPermission()) {
      throw StateError('Microphone permission is required.');
    }

    _isRecording = true;
    StreamSubscription<Uint8List>? sub;
    try {
      for (var s = countdown.inSeconds; s > 0; s--) {
        onStatus?.call('Get ready... $s');
        await Future<void>.delayed(const Duration(seconds: 1));
      }

      final pcm = BytesBuilder(copy: false);
      final stream = await _recorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: _sampleRate,
        numChannels: 1,
      ));
      sub = stream.listen(pcm.add);

      for (var s = duration.inSeconds; s > 0; s--) {
        onStatus?.call('Recording ${label.name}... ${s}s left');
        await Future<void>.delayed(const Duration(seconds: 1));
      }

      await sub.cancel();
      sub = null;
      if (await _recorder.isRecording()) {
        await _recorder.stop();
      }

      final base = await outputDirectory();
      final labelDir = Directory('${base.path}/${label.name}');
      if (!await labelDir.exists()) await labelDir.create(recursive: true);
      final stamp = DateTime.now().millisecondsSinceEpoch;
      final file = File('${labelDir.path}/${label.name}_$stamp.wav');
      await file.writeAsBytes(_wrapAsWav(pcm.takeBytes(), _sampleRate));
      onStatus?.call('Saved ${file.uri.pathSegments.last}');
      return file.path;
    } finally {
      await sub?.cancel();
      _isRecording = false;
    }
  }

  /// Wraps raw 16-bit mono little-endian PCM in a canonical 44-byte WAV
  /// header.
  static Uint8List _wrapAsWav(Uint8List pcm, int sampleRate) {
    final b = BytesBuilder();
    void w32(int v) =>
        b.add((ByteData(4)..setUint32(0, v, Endian.little)).buffer.asUint8List());
    void w16(int v) =>
        b.add((ByteData(2)..setUint16(0, v, Endian.little)).buffer.asUint8List());

    b.add('RIFF'.codeUnits);
    w32(36 + pcm.length);
    b.add('WAVE'.codeUnits);
    b.add('fmt '.codeUnits);
    w32(16); // fmt chunk size
    w16(1); // PCM
    w16(1); // mono
    w32(sampleRate);
    w32(sampleRate * 2); // byte rate
    w16(2); // block align
    w16(16); // bits per sample
    b.add('data'.codeUnits);
    w32(pcm.length);
    b.add(pcm);
    return b.toBytes();
  }
}
