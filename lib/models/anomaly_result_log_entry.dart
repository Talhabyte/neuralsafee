import '../services/anomaly_evaluator.dart';
import 'behavioral_classification_result.dart';

/// Persistence-layer wrapper around a pure AnomalyResult. AnomalyResult
/// itself carries no timestamp or window-key context (it's a pure
/// computation output, deliberately kept that way) — this class adds
/// exactly the metadata a stored log entry needs, without modifying
/// AnomalyResult itself.
///
/// [classification] is the Random Forest + linear SVM ensemble's
/// output for this entry's z-scores (NeuralSafe proposal Section 6.3's
/// classification layer on top of AnomalyEvaluator), and is additive:
/// null whenever [result].status isn't AnomalyStatus.evaluated, and
/// also null for any entry persisted before this field existed —
/// toMap()/fromMap() treat it as optional so old stored entries keep
/// loading correctly with classification simply absent.
class AnomalyResultLogEntry {
  final DateTime evaluatedAt;
  final String windowKey;
  final AnomalyResult result;
  final BehavioralClassificationResult? classification;

  const AnomalyResultLogEntry({
    required this.evaluatedAt,
    required this.windowKey,
    required this.result,
    this.classification,
  });

  Map<String, dynamic> toMap() {
    return {
      'evaluatedAt': evaluatedAt.millisecondsSinceEpoch,
      'windowKey': windowKey,
      'status': result.status.name,
      'aggregateScore': result.aggregateScore,
      'sampleCountUsed': result.sampleCountUsed,
      'featureZScores': result.featureZScores,
      if (classification != null) 'classification': classification!.toMap(),
    };
  }

  factory AnomalyResultLogEntry.fromMap(Map<dynamic, dynamic> map) {
    final rawEvaluatedAt = map['evaluatedAt'] as int?;
    final rawStatus = map['status'] as String?;
    final rawFeatureZScores = map['featureZScores'];
    final rawClassification = map['classification'];

    final status = switch (rawStatus) {
      'coldStart' => AnomalyStatus.coldStart,
      'evaluated' => AnomalyStatus.evaluated,
      _ => throw ArgumentError('Unknown AnomalyStatus: $rawStatus'),
    };

    final featureZScores = <String, double>{};
    if (rawFeatureZScores is Map) {
      rawFeatureZScores.forEach((key, value) {
        featureZScores[key as String] = (value as num).toDouble();
      });
    }

    return AnomalyResultLogEntry(
      evaluatedAt: rawEvaluatedAt != null
          ? DateTime.fromMillisecondsSinceEpoch(rawEvaluatedAt)
          : DateTime.now(),
      windowKey: map['windowKey'] as String,
      result: AnomalyResult(
        status: status,
        featureZScores: featureZScores,
        aggregateScore: (map['aggregateScore'] as num).toDouble(),
        sampleCountUsed: map['sampleCountUsed'] as int,
      ),
      classification: BehavioralClassificationResult.fromMapOrNull(
        rawClassification is Map ? rawClassification : null,
      ),
    );
  }

  @override
  String toString() {
    return 'AnomalyResultLogEntry(evaluatedAt: $evaluatedAt, '
        'windowKey: $windowKey, result: $result, '
        'classification: $classification)';
  }
}
