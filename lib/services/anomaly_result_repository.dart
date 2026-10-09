import '../models/anomaly_result_log_entry.dart';
import '../models/behavioral_classification_result.dart';
import '../services/anomaly_evaluator.dart';
import 'feature_vault_service.dart';

/// Persists AnomalyResultLogEntry records to FeatureVaultService's box,
/// under a dedicated 'anomaly_result_log' key holding a List of Maps.
/// Rolling cap of 500 entries (matching BehavioralEventRepository's
/// precedent), oldest entries trimmed first, chronological order
/// preserved — same pattern as BehavioralEventRepository.save().
class AnomalyResultRepository {
  static const _logKey = 'anomaly_result_log';
  static const int maxEntries = 500;

  List<AnomalyResultLogEntry> loadAll() {
    final raw = FeatureVaultService.instance.box.get(_logKey);
    if (raw is! List) return [];

    return raw
        .whereType<Map>()
        .map((entry) => AnomalyResultLogEntry.fromMap(entry))
        .toList();
  }

  /// [classification] is optional — stored alongside [result] when the
  /// Behavioral classifier produced one, left absent otherwise (cold
  /// start, or the classifier asset failed to load). See
  /// AnomalyResultLogEntry's doc comment for the persisted shape.
  Future<void> save(
    AnomalyResult result, {
    required DateTime evaluatedAt,
    required String windowKey,
    BehavioralClassificationResult? classification,
  }) async {
    final entry = AnomalyResultLogEntry(
      evaluatedAt: evaluatedAt,
      windowKey: windowKey,
      result: result,
      classification: classification,
    );

    final current = loadAll()..add(entry);

    final trimmed = current.length > maxEntries
        ? current.sublist(current.length - maxEntries)
        : current;

    await FeatureVaultService.instance.box.put(
      _logKey,
      trimmed.map((e) => e.toMap()).toList(),
    );
  }

  Future<void> clearAll() async {
    await FeatureVaultService.instance.box.put(_logKey, []);
  }
}
