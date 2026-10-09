import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../models/behavioral_classification_result.dart';

/// Loads and runs the trained Random Forest + linear SVM ensemble
/// (assets/ml/behavioral_classifier_model.json, trained offline by
/// tools/behavioral_model/generate_and_train.py on a synthetic,
/// literature-grounded dataset — see that script's docstring and the
/// thesis methodology/evaluation chapters for why no real dataset is
/// used) against the 11-dim z-score vector AnomalyEvaluator produces.
///
/// This is the classification layer NeuralSafe's own approved proposal
/// (Section 6.3) describes: "Z-score deviations are fed to an SVM
/// classifier producing three output classes." AnomalyEvaluator is the
/// correct input stage for that and is unchanged by this class — this
/// only adds the missing classification step on top of its output.
///
/// Implemented as a plain Dart tree-walking / dot-product interpreter
/// rather than a TFLite model: scikit-learn's RandomForestClassifier
/// and LinearSVC don't export to TFLite directly, and this ensemble is
/// small enough (40 shallow trees + one 11x3 linear layer, ~340KB of
/// JSON) that a hand-written interpreter is simpler than standing up a
/// conversion pipeline, while keeping everything on-device with no new
/// native dependency.
class BehavioralClassifier {
  BehavioralClassifier._(this._model)
      : featureKeys = List<String>.from(_model['featureKeys'] as List),
        classes = List<String>.from(_model['classes'] as List),
        _rfWeight = ((_model['ensembleWeights'] as Map)['randomForest'] as num)
            .toDouble(),
        _svmWeight = ((_model['ensembleWeights'] as Map)['linearSvm'] as num)
            .toDouble(),
        _scoreBandMidpoints = (_model['scoreBandMidpoints'] as Map).map(
          (key, value) => MapEntry(key as String, (value as num).toDouble()),
        ),
        _trees = (_model['randomForest'] as Map)['trees'] as List,
        _svmCoef = ((_model['linearSvm'] as Map)['coef'] as List)
            .map((row) => (row as List)
                .map((v) => (v as num).toDouble())
                .toList(growable: false))
            .toList(growable: false),
        _svmIntercept = ((_model['linearSvm'] as Map)['intercept'] as List)
            .map((v) => (v as num).toDouble())
            .toList(growable: false);

  static const String assetPath =
      'assets/ml/behavioral_classifier_model.json';

  static BehavioralClassifier? _instance;
  static Future<BehavioralClassifier>? _loading;

  final Map<String, dynamic> _model;

  /// Feature order the model was trained on. Must stay in sync with
  /// lib/models/behavioral_baseline.dart's trackedFeatureKeys — both
  /// lists are byte-for-byte identical today (see
  /// tools/behavioral_model/generate_and_train.py's docstring).
  final List<String> featureKeys;

  /// Class label order, consistently used across randomForest.classes,
  /// linearSvm.classes, and scoreBandMidpoints' keys in the model file.
  final List<String> classes;

  final double _rfWeight;
  final double _svmWeight;
  final Map<String, double> _scoreBandMidpoints;
  final List<dynamic> _trees;
  final List<List<double>> _svmCoef;
  final List<double> _svmIntercept;

  /// Application-level singleton. Loads and parses the asset the first
  /// time it's needed, then reuses the same parsed model on every later
  /// call — the JSON is ~340KB, not something to re-parse per tick.
  static Future<BehavioralClassifier> instance() async {
    if (_instance != null) return _instance!;
    _loading ??= _load();
    _instance = await _loading;
    return _instance!;
  }

  static Future<BehavioralClassifier> _load() async {
    final raw = await rootBundle.loadString(assetPath);
    final model = jsonDecode(raw) as Map<String, dynamic>;
    return BehavioralClassifier._(model);
  }

  /// Test-only construction path, bypassing the asset bundle (rootBundle
  /// isn't available outside a running Flutter app). Takes an
  /// already-decoded model map, e.g. loaded from a fixture file or
  /// built in-memory for a unit test.
  @visibleForTesting
  factory BehavioralClassifier.fromModel(Map<String, dynamic> model) {
    return BehavioralClassifier._(model);
  }

  /// Classifies a feature-key -> z-score map (AnomalyResult's
  /// featureZScores) into Normal / MildlyAnomalous / CriticallyAnomalous.
  /// A key missing from [featureZScores] defaults to 0.0 ("no observed
  /// deviation") rather than throwing, so a partial map degrades
  /// gracefully instead of crashing the pipeline.
  BehavioralClassificationResult classify(
    Map<String, double> featureZScores,
  ) {
    final vector = featureKeys
        .map((key) => featureZScores[key] ?? 0.0)
        .toList(growable: false);

    final rfProba = _randomForestProba(vector);
    final svmProba = _softmax(_svmDecisionScores(vector));

    final ensembleProba = List<double>.generate(
      classes.length,
      (i) => _rfWeight * rfProba[i] + _svmWeight * svmProba[i],
    );

    var bestIndex = 0;
    for (var i = 1; i < ensembleProba.length; i++) {
      if (ensembleProba[i] > ensembleProba[bestIndex]) bestIndex = i;
    }

    final classProbabilities = <String, double>{
      for (var i = 0; i < classes.length; i++) classes[i]: ensembleProba[i],
    };

    return BehavioralClassificationResult(
      predictedClass: classes[bestIndex],
      classProbabilities: classProbabilities,
      behaviorScore: _weightedScore(classProbabilities),
    );
  }

  double _weightedScore(Map<String, double> classProbabilities) {
    var score = 0.0;
    classProbabilities.forEach((className, prob) {
      score += prob * (_scoreBandMidpoints[className] ?? 0.0);
    });
    return score.clamp(0.0, 100.0);
  }

  List<double> _randomForestProba(List<double> vector) {
    final sum = List<double>.filled(classes.length, 0.0);
    for (final tree in _trees) {
      final leafProbs = _walkTree(tree as Map<String, dynamic>, vector);
      for (var i = 0; i < sum.length; i++) {
        sum[i] += leafProbs[i];
      }
    }
    final count = _trees.length;
    if (count == 0) return sum;
    return sum.map((v) => v / count).toList(growable: false);
  }

  /// Walks a single exported tree. Node shape mirrors sklearn's
  /// tree_ layout exactly (children_left / children_right / feature /
  /// threshold), including sklearn's branch convention: a sample goes
  /// left when vector[featureIndex] <= threshold, right otherwise. A
  /// leaf node carries the full class-probability vector for that leaf
  /// (not just the argmax), so the forest-wide average is a true
  /// probability average, not a majority vote.
  List<double> _walkTree(Map<String, dynamic> node, List<double> vector) {
    if (node['leaf'] == true) {
      return (node['probs'] as List)
          .map((v) => (v as num).toDouble())
          .toList(growable: false);
    }
    final featureIndex = node['featureIndex'] as int;
    final threshold = (node['threshold'] as num).toDouble();
    final next =
        vector[featureIndex] <= threshold ? node['left'] : node['right'];
    return _walkTree(next as Map<String, dynamic>, vector);
  }

  /// LinearSVC with multi_class='ovr': decision for class c is
  /// coef[c]·vector + intercept[c]. Matches
  /// generate_and_train.py::export_linear_svm exactly.
  List<double> _svmDecisionScores(List<double> vector) {
    return List<double>.generate(classes.length, (classIndex) {
      final coefRow = _svmCoef[classIndex];
      var dot = 0.0;
      for (var i = 0; i < vector.length; i++) {
        dot += coefRow[i] * vector[i];
      }
      return dot + _svmIntercept[classIndex];
    });
  }

  List<double> _softmax(List<double> scores) {
    final maxScore = scores.reduce(max);
    final expScores = scores.map((s) => exp(s - maxScore)).toList();
    final sumExp = expScores.reduce((a, b) => a + b);
    return expScores.map((e) => e / sumExp).toList(growable: false);
  }
}
