/// Pure data holder for the Behavioral module's Random Forest + linear
/// SVM ensemble output (NeuralSafe proposal Section 6.3: "Z-score
/// deviations are fed to an SVM classifier producing three output
/// classes"). AnomalyEvaluator still computes the per-feature z-scores
/// this is built from — this class is the classification layer sitting
/// on top of it, not a replacement for it.
class BehavioralClassificationResult {
  /// One of BehavioralClassifier.classes ("Normal", "MildlyAnomalous",
  /// "CriticallyAnomalous") — the ensemble's argmax prediction.
  final String predictedClass;

  /// Ensemble class probabilities (0.5 * RandomForest.predict_proba +
  /// 0.5 * softmax(LinearSVM.decision_function), weights as recorded in
  /// the model file), keyed by class name, summing to ~1.0.
  final Map<String, double> classProbabilities;

  /// Continuous 0-100 score: a probability-weighted average of each
  /// class's scoreBandMidpoint. Smoother than the discrete predicted
  /// class alone, so DangerScoreFusionEngine receives a continuous
  /// BehaviorScore rather than one of three fixed values.
  final double behaviorScore;

  const BehavioralClassificationResult({
    required this.predictedClass,
    required this.classProbabilities,
    required this.behaviorScore,
  });

  Map<String, dynamic> toMap() {
    return {
      'predictedClass': predictedClass,
      'classProbabilities': classProbabilities,
      'behaviorScore': behaviorScore,
    };
  }

  /// Returns null if [map] is null — callers use this to keep
  /// classification optional in persisted records (older entries saved
  /// before this field existed have no 'classification' key at all).
  static BehavioralClassificationResult? fromMapOrNull(
    Map<dynamic, dynamic>? map,
  ) {
    if (map == null) return null;

    final rawProbs = map['classProbabilities'];
    final probs = <String, double>{};
    if (rawProbs is Map) {
      rawProbs.forEach((key, value) {
        probs[key as String] = (value as num).toDouble();
      });
    }

    return BehavioralClassificationResult(
      predictedClass: map['predictedClass'] as String,
      classProbabilities: probs,
      behaviorScore: (map['behaviorScore'] as num).toDouble(),
    );
  }

  @override
  String toString() {
    return 'BehavioralClassificationResult(predictedClass: $predictedClass, '
        'behaviorScore: $behaviorScore, '
        'classProbabilities: $classProbabilities)';
  }
}
