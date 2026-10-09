import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/anomaly_result_log_entry.dart';
import '../models/behavioral_baseline.dart';
import '../models/behavioral_classification_result.dart';
import 'anomaly_evaluator.dart';
import 'anomaly_result_notifier.dart';
import 'behavioral_baseline_repository.dart';
import 'behavioral_classifier.dart';
import 'feature_extractor.dart';

/// Closes the gap BehavioralMonitoringService's own aggregator comment
/// flags ("nothing acts on the batches yet"): on a fixed period, this
/// computes a BehavioralFeatureVector over a trailing window ending
/// now, evaluates it against the persisted running baseline for
/// [windowKey] via AnomalyEvaluator, runs the Random Forest + linear
/// SVM ensemble (BehavioralClassifier) over the resulting z-scores —
/// the classification layer NeuralSafe proposal Section 6.3 describes
/// — records both via AnomalyResultNotifier (persist + emit on
/// [anomalyResults]), and finally updates+persists the baseline with
/// the new sample.
///
/// Evaluation happens BEFORE the baseline is updated with the new
/// sample, so a baseline with sampleCount == N is what the (N+1)th
/// tick evaluates against — the Nth tick is the one that pushes
/// sampleCount to N. With AnomalyEvaluator's default minSampleCount
/// of 10, the 11th tick is the first to produce AnomalyStatus.evaluated
/// rather than coldStart, and therefore the first tick the classifier
/// runs on (classification stays null during coldStart — there are no
/// z-scores yet to classify).
///
/// One pipeline instance = one window size/key. Every dependency is
/// injected as a plain function (not a whole repository object),
/// matching this codebase's existing pattern (FeatureExtractor's
/// loadEvents, AnomalyResultNotifier's saveEntry, etc.) so this is
/// fully unit-testable without touching the encrypted vault or the
/// classifier's asset bundle.
class BehavioralAnomalyPipeline {
  BehavioralAnomalyPipeline._({
    required this.windowKey,
    required this.windowSize,
    required FeatureExtractor featureExtractor,
    required BehavioralBaseline? Function(String windowKey) loadBaseline,
    required Future<void> Function(BehavioralBaseline baseline) saveBaseline,
    required AnomalyEvaluator evaluator,
    required AnomalyResultNotifier notifier,
    required Future<BehavioralClassificationResult?> Function(
      Map<String, double> featureZScores,
    )
        classify,
    required DateTime Function() clock,
  })  : _featureExtractor = featureExtractor,
        _loadBaseline = loadBaseline,
        _saveBaseline = saveBaseline,
        _evaluator = evaluator,
        _notifier = notifier,
        _classify = classify,
        _clock = clock;

  static BehavioralAnomalyPipeline? _instance;

  /// Deliberately short (2 minutes) so a live demo can accumulate
  /// AnomalyEvaluator's default minSampleCount (10) worth of samples
  /// in ~22 minutes rather than multiple hours. Trade-off: a 2-minute
  /// behavioral window is noisier/less statistically meaningful than
  /// the "15m"/"1h"/"6h"/"24h" windows BehavioralBaseline's own doc
  /// comment uses as examples. Revisit both constants together for a
  /// real longer-horizon deployment once demo timing isn't a
  /// constraint.
  static const String _defaultWindowKey = '2m';
  static const Duration _defaultWindowSize = Duration(minutes: 2);

  /// Application-level singleton, using the real encrypted-vault-backed
  /// BehavioralBaselineRepository, a real AnomalyResultNotifier, and
  /// the real asset-backed BehavioralClassifier.
  static BehavioralAnomalyPipeline get instance {
    if (_instance != null) return _instance!;
    final repository = BehavioralBaselineRepository();
    _instance = BehavioralAnomalyPipeline._(
      windowKey: _defaultWindowKey,
      windowSize: _defaultWindowSize,
      featureExtractor: FeatureExtractor(),
      loadBaseline: repository.load,
      saveBaseline: repository.save,
      // TEMPORARY (revert to const AnomalyEvaluator() — minSampleCount
      // defaults to 10 — once the Behavioral classifier is confirmed
      // working on-device): minSampleCount: 2 means the 3rd tick (~6
      // minutes of monitoring) is already "evaluated" instead of
      // needing 11 ticks (~22 minutes), so you don't have to sit
      // through a long wait just to see the classifier run once.
      evaluator: const AnomalyEvaluator(minSampleCount: 2),
      notifier: AnomalyResultNotifier(),
      classify: _defaultClassify,
      clock: DateTime.now,
    );
    return _instance!;
  }

  /// Runs the real BehavioralClassifier singleton. Any failure (asset
  /// missing or corrupt, bad JSON, a malformed vector) degrades
  /// gracefully to null rather than throwing out of [tick] — a tick
  /// still records the z-score AnomalyResult either way, and
  /// RiskDecisionEngine falls back to AnomalyResult.aggregateScore
  /// whenever classification is null.
  static Future<BehavioralClassificationResult?> _defaultClassify(
    Map<String, double> featureZScores,
  ) async {
    try {
      final classifier = await BehavioralClassifier.instance();
      return classifier.classify(featureZScores);
    } catch (e) {
      // TEMPORARY (remove once verified on-device): makes a bad/missing
      // model asset visible in `flutter run` output instead of silently
      // falling back to AnomalyResult.aggregateScore.
      debugPrint('[NeuralSafe] Behavioral classifier failed to run: $e');
      return null;
    }
  }

  /// Test-only construction path. Every optional parameter defaults to
  /// a lightweight in-memory stand-in (no vault, no asset bundle, no
  /// real clock), mirroring BehavioralMonitoringService.test()'s
  /// pattern.
  @visibleForTesting
  factory BehavioralAnomalyPipeline.test({
    required String windowKey,
    required Duration windowSize,
    FeatureExtractor? featureExtractor,
    BehavioralBaseline? Function(String windowKey)? loadBaseline,
    Future<void> Function(BehavioralBaseline baseline)? saveBaseline,
    AnomalyEvaluator? evaluator,
    AnomalyResultNotifier? notifier,
    Future<BehavioralClassificationResult?> Function(
      Map<String, double> featureZScores,
    )?
        classify,
    DateTime Function()? clock,
  }) {
    return BehavioralAnomalyPipeline._(
      windowKey: windowKey,
      windowSize: windowSize,
      featureExtractor: featureExtractor ?? FeatureExtractor(),
      loadBaseline: loadBaseline ?? (_) => null,
      saveBaseline: saveBaseline ?? (_) async {},
      evaluator: evaluator ?? const AnomalyEvaluator(),
      notifier: notifier ?? AnomalyResultNotifier(),
      classify: classify ?? (_) async => null,
      clock: clock ?? DateTime.now,
    );
  }

  final String windowKey;
  final Duration windowSize;
  final FeatureExtractor _featureExtractor;
  final BehavioralBaseline? Function(String windowKey) _loadBaseline;
  final Future<void> Function(BehavioralBaseline baseline) _saveBaseline;
  final AnomalyEvaluator _evaluator;
  final AnomalyResultNotifier _notifier;
  final Future<BehavioralClassificationResult?> Function(
    Map<String, double> featureZScores,
  ) _classify;
  final DateTime Function() _clock;

  Timer? _timer;
  bool get isRunning => _timer != null;

  /// Feed this directly into RiskDecisionEngine as its anomalyStream.
  Stream<AnomalyResultLogEntry> get anomalyResults => _notifier.history;
  AnomalyResultLogEntry? get latest => _notifier.latest;

  /// Starts periodic evaluation. Idempotent.
  void start() {
    if (_timer != null) return;
    _timer = Timer.periodic(windowSize, (_) => tick());
  }

  /// Stops periodic evaluation. Safe to call even if not running.
  void stop() {
    _timer?.cancel();
    _timer = null;
  }

  /// Runs a single evaluate-classify-and-update cycle. Exposed publicly
  /// (not just via the internal Timer) so tests can drive it
  /// deterministically, and so a manual "evaluate now" trigger is
  /// possible later if wanted (e.g. a demo button).
  Future<void> tick() async {
    final end = _clock();
    final start = end.subtract(windowSize);
    final vector = _featureExtractor.extractWindow(start, end);

    final baseline =
        _loadBaseline(windowKey) ?? BehavioralBaseline.initial(windowKey);

    final result = _evaluator.evaluate(vector, baseline);

    final classification = result.status == AnomalyStatus.evaluated
        ? await _classify(result.featureZScores)
        : null;

    // TEMPORARY (remove once verified on-device): proves the classifier
    // actually ran and what it produced, since classification is
    // otherwise invisible until something reaches highDanger.
    if (result.status == AnomalyStatus.evaluated) {
      debugPrint(
        '[NeuralSafe] Behavioral z-score evaluation: '
        'aggregateScore=${result.aggregateScore.toStringAsFixed(1)}',
      );
      if (classification != null) {
        debugPrint(
          '[NeuralSafe] Behavioral classifier: ${classification.predictedClass} '
          '(behaviorScore=${classification.behaviorScore.toStringAsFixed(1)}, '
          'probs=${classification.classProbabilities})',
        );
      } else {
        debugPrint(
          '[NeuralSafe] Behavioral classifier produced no result this tick '
          '— falling back to aggregateScore (see warning above if the '
          'asset failed to load).',
        );
      }
    }

    await _notifier.record(
      result,
      windowKey: windowKey,
      classification: classification,
    );

    final updatedBaseline = baseline.updateWith(vector);
    await _saveBaseline(updatedBaseline);
  }
}
