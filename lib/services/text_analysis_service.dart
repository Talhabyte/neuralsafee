import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

import 'text_rules.dart';

class TextAnalysisResult {
  final double score; // 0-100
  final String predictedClass;
  final List<double> probabilities;

  TextAnalysisResult({
    required this.score,
    required this.predictedClass,
    required this.probabilities,
  });

  @override
  String toString() =>
      'TextAnalysisResult(score: $score, class: $predictedClass, probs: $probabilities)';
}

// LEGACY keyword lists (substring matching; no longer used for scoring).
// Scoring now uses TextRules in text_rules.dart, which matches whole words,
// handles negation / idioms / Roman Urdu, and detects short pleas.
const List<String> kHighRiskWords = [
  'kill',
  'murder',
  'gun',
  'knife',
  'weapon',
  'die',
  'dead',
  'stab',
  'strangle',
  'suicide',
  'rape',
  'مار',
  'قتل',
  'خودکشی',
  'بندوق',
  'چھری',
];

const List<String> kModerateRiskWords = [
  'hit',
  'hurt',
  'scream',
  'afraid',
  'scared',
  'threat',
  'threaten',
  'help',
  'danger',
  'abuse',
  'beat',
  'hostage',
  'trapped',
  'ڈر',
  'خوف',
  'مدد',
  'دھمکی',
];

/// Pure, unit-testable scoring function per FYP Chapter 4.5.2 pseudocode.
TextAnalysisResult computeTextScore(
    String originalText, List<double> classProbs) {
  assert(classProbs.length == 4);
  final pNormal = classProbs[0];
  final pDistress = classProbs[1];
  final pViolence = classProbs[2];
  final pHelp = classProbs[3];

  const classNames = ['Normal', 'Distress', 'Violence', 'Help'];
  var maxIdx = 0;
  var maxP = classProbs[0];
  for (var i = 1; i < classProbs.length; i++) {
    if (classProbs[i] > maxP) {
      maxP = classProbs[i];
      maxIdx = i;
    }
  }
  final predictedClass = classNames[maxIdx];

  // Rule layer: idioms / negation / fiction => benign; short pleas, Roman Urdu,
  // coercive-control phrases etc. => minimum score ("floor").
  final verdict = TextRules.evaluate(originalText);
  if (verdict.benign && verdict.floor <= 0) {
    return TextAnalysisResult(
      score: 5.0,
      predictedClass: 'Normal',
      probabilities: classProbs,
    );
  }

  double baseScore;
  switch (predictedClass) {
    case 'Violence':
      baseScore = 80 + pViolence * 20;
      break;
    case 'Distress':
      baseScore = 50 + pDistress * 30;
      break;
    case 'Help':
      baseScore = 40 + pHelp * 40;
      break;
    default:
      baseScore = pNormal * 20;
  }

  double modelScore;
  if (pViolence > 0.8 || pDistress > 0.75 || pHelp > 0.75) {
    modelScore = baseScore;
  } else {
    final maxRiskProb =
        [pViolence, pDistress, pHelp].reduce((a, b) => a > b ? a : b);
    modelScore = baseScore * (maxRiskProb / 0.8);
  }

  var finalScore = modelScore;
  var finalClass = predictedClass;

  // Uncertain model output (no class above 50%) with no rule firing is noise
  // (e.g. everyday sentences the model has never seen): treat as no signal.
  if (maxP < 0.5 && verdict.floor <= 0) {
    finalScore = modelScore > 15 ? 15.0 : modelScore;
    finalClass = 'Normal';
  }

  if (verdict.floor > finalScore) {
    finalScore = verdict.floor;
    if (verdict.cls != null) finalClass = verdict.cls!;
  }
  final capped = finalScore > 100 ? 100.0 : finalScore;

  return TextAnalysisResult(
    score: capped,
    predictedClass: finalClass,
    probabilities: classProbs,
  );
}

class TextAnalysisService {
  TextAnalysisService._();
  static final TextAnalysisService instance = TextAnalysisService._();

  SentencePieceTokenizer? _tokenizer;
  Interpreter? _interpreter;
  bool _initialized = false;

  static const int _maxLen = 128;
  static const int _padId = 1;

  bool get isInitialized => _initialized;

  Future<void> init() async {
    if (_initialized) return;

    final bytes = await rootBundle.load('assets/ml/sentencepiece.bpe.model');
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/sentencepiece.bpe.model');
    await file.writeAsBytes(bytes.buffer.asUint8List());

    _tokenizer = SentencePieceTokenizer.fromModelFileSync(file.path);
    _interpreter =
        await Interpreter.fromAsset('assets/ml/neuralsafe_text_model.tflite');
    _initialized = true;
  }

  int _toXlmrId(int rawId) {
    if (rawId == 0) return 3; // <unk>
    if (rawId == 1) return 0; // <s>
    if (rawId == 2) return 2; // </s>
    return rawId + 1;
  }

  Future<List<double>> _inferProbs(String text) async {
    final rawIds = _tokenizer!.encode(text).ids;
    final contentIds = rawIds.map(_toXlmrId).toList();
    var fullIds = [0, ...contentIds, 2];

    if (fullIds.length > _maxLen) {
      fullIds = fullIds.sublist(0, _maxLen - 1) + [2];
    }

    final attentionMask = List<int>.filled(_maxLen, 0);
    final inputIds = List<int>.filled(_maxLen, _padId);
    for (var i = 0; i < fullIds.length; i++) {
      inputIds[i] = fullIds[i];
      attentionMask[i] = 1;
    }

    final output = List.filled(1 * 4, 0.0).reshape([1, 4]);
    _interpreter!.runForMultipleInputs(
      [
        [attentionMask],
        [inputIds],
      ],
      {0: output},
    );

    return (output[0] as List).map((e) => e as double).toList();
  }

  /// Scores each sentence separately (max 12) and returns the most dangerous
  /// one, so a dangerous line cannot be diluted by harmless text around it or
  /// cut off by the model's 128-token limit.
  Future<TextAnalysisResult> analyze(String text) async {
    if (!_initialized) {
      await init();
    }

    final segments = TextRules.splitSegments(text);
    if (segments.isEmpty) {
      return computeTextScore(text, await _inferProbs(text));
    }

    TextAnalysisResult? best;
    for (final seg in segments.take(12)) {
      final r = computeTextScore(seg, await _inferProbs(seg));
      if (best == null || r.score > best.score) best = r;
    }
    return best!;
  }
}
