import 'dart:async';

import '../models/anomaly_result_log_entry.dart';
import '../models/behavioral_classification_result.dart';
import '../services/anomaly_evaluator.dart';
import 'anomaly_result_repository.dart';

/// Consumes AnomalyResult outputs, persists them, and exposes the
/// latest result and a stream of history — WITHOUT any threshold
/// comparison or alert-triggering logic. This is deliberately neutral
/// plumbing: "here is what was evaluated and when," not "here is
/// whether something is wrong." Threshold/alert decision logic is
/// explicitly out of scope for this step and lives in
/// RiskDecisionEngine/DangerScoreFusionEngine instead.
///
/// [classification] is an optional pass-through for the Random
/// Forest + linear SVM ensemble's output on the same z-scores
/// (BehavioralClassifier, invoked by BehavioralAnomalyPipeline) — this
/// class still does no interpretation of it, just stores and forwards
/// it alongside the raw AnomalyResult it was computed from.
///
/// [saveEntry] is injectable (defaults to
/// AnomalyResultRepository().save) so this can be unit tested without
/// the encrypted vault.
class AnomalyResultNotifier {
  AnomalyResultNotifier({
    Future<void> Function(
      AnomalyResult result, {
      required DateTime evaluatedAt,
      required String windowKey,
      BehavioralClassificationResult? classification,
    })? saveEntry,
    DateTime Function()? clock,
  })  : _saveEntry = saveEntry ?? AnomalyResultRepository().save,
        _clock = clock ?? DateTime.now,
        _controller = StreamController<AnomalyResultLogEntry>.broadcast();

  final Future<void> Function(
    AnomalyResult result, {
    required DateTime evaluatedAt,
    required String windowKey,
    BehavioralClassificationResult? classification,
  }) _saveEntry;
  final DateTime Function() _clock;
  final StreamController<AnomalyResultLogEntry> _controller;

  AnomalyResultLogEntry? _latest;

  /// The most recent entry recorded via [record], or null if none yet.
  AnomalyResultLogEntry? get latest => _latest;

  /// Broadcast stream of every recorded entry, in the order recorded.
  Stream<AnomalyResultLogEntry> get history => _controller.stream;

  /// Records a new AnomalyResult: persists it via [saveEntry], updates
  /// [latest], and emits it on [history]. Performs NO comparison
  /// against any threshold and takes NO action based on the result's
  /// status, score, or [classification] beyond storing and exposing it
  /// as-is.
  Future<void> record(
    AnomalyResult result, {
    required String windowKey,
    BehavioralClassificationResult? classification,
  }) async {
    final evaluatedAt = _clock();

    await _saveEntry(
      result,
      evaluatedAt: evaluatedAt,
      windowKey: windowKey,
      classification: classification,
    );

    final entry = AnomalyResultLogEntry(
      evaluatedAt: evaluatedAt,
      windowKey: windowKey,
      result: result,
      classification: classification,
    );

    _latest = entry;
    if (!_controller.isClosed) {
      _controller.add(entry);
    }
  }

  /// Permanently closes the history stream.
  Future<void> dispose() async {
    await _controller.close();
  }
}
