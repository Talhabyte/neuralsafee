import 'dart:async';

import '../models/anomaly_result_log_entry.dart';
import '../models/risk_decision.dart';
import '../models/risk_tier.dart';
import 'anomaly_evaluator.dart';
import 'behavioral_anomaly_pipeline.dart';
import 'danger_score_fusion_engine.dart';
import 'sustained_danger_tracker.dart';

class RiskDecisionEngine {
  RiskDecisionEngine({
    required Stream<AnomalyResultLogEntry> anomalyStream,
    DangerScoreFusionEngine? fusionEngine,
    SustainedDangerTracker? sustainedDangerTracker,
  })  : _anomalyStream = anomalyStream,
        _fusionEngine = fusionEngine ?? const DangerScoreFusionEngine(),
        _sustainedTracker = sustainedDangerTracker ?? SustainedDangerTracker(),
        _controller = StreamController<RiskDecision>.broadcast();

  static RiskDecisionEngine? _instance;

  /// Application-level singleton, wired to the real, persistent
  /// BehavioralAnomalyPipeline (rather than any test-only or
  /// dashboard-local stream). Constructed and started lazily on first
  /// access — main.dart accesses this once at startup specifically to
  /// trigger that construction, and every other part of the app
  /// (dashboard, future MessageInterceptor, etc.) should reference
  /// this same instance rather than constructing its own.
  static RiskDecisionEngine get instance {
    if (_instance != null) return _instance!;
    _instance = RiskDecisionEngine(
      anomalyStream: BehavioralAnomalyPipeline.instance.anomalyResults,
    )..start();
    return _instance!;
  }

  final Stream<AnomalyResultLogEntry> _anomalyStream;
  final DangerScoreFusionEngine _fusionEngine;

  /// Implements FR-4.2's sustained-danger boost: tracks whether the
  /// RAW fused score has stayed continuously at/above 60 for 5+
  /// seconds, and if so, adds +20 before classification — see that
  /// class's doc comment for why this lives here rather than inside
  /// DangerScoreFusionEngine itself.
  final SustainedDangerTracker _sustainedTracker;

  final StreamController<RiskDecision> _controller;

  StreamSubscription<AnomalyResultLogEntry>? _subscription;

  bool get isRunning => _subscription != null;

  RiskDecision? _latest;
  RiskDecision? get latest => _latest;

  double? _latestBehaviorScore;
  double? _latestTextScore;
  double? _latestVoiceScore;

  Stream<RiskDecision> get decisions => _controller.stream;

  void start() {
    if (_subscription != null) return;
    _subscription = _anomalyStream.listen(_handleAnomalyEntry);
  }

  Future<void> stop() async {
    await _subscription?.cancel();
    _subscription = null;
    // So a later restart's first decision isn't boosted off a
    // sustained period that happened before this stop.
    _sustainedTracker.reset();
  }

  Future<void> dispose() async {
    await _controller.close();
  }

  void submitTextScore(
    double score, {
    DateTime? evaluatedAt,
    String windowKey = 'text-event',
  }) {
    _latestTextScore = score;
    _emitDecision(
      evaluatedAt: evaluatedAt ?? DateTime.now(),
      windowKey: windowKey,
    );
  }

  void submitVoiceScore(
    double score, {
    DateTime? evaluatedAt,
    String windowKey = 'voice-event',
  }) {
    _latestVoiceScore = score;
    _emitDecision(
      evaluatedAt: evaluatedAt ?? DateTime.now(),
      windowKey: windowKey,
    );
  }

  void _handleAnomalyEntry(AnomalyResultLogEntry entry) {
    // Prefer the Random Forest + linear SVM ensemble's continuous
    // BehaviorScore (NeuralSafe proposal Section 6.3's classification
    // layer) whenever the classifier produced one for this entry.
    // Falls back to AnomalyEvaluator's own aggregateScore — the
    // pre-classifier z-score signal — when classification is
    // unavailable (classifier asset missing/corrupt) but the z-score
    // evaluation itself is still evaluated rather than coldStart, so a
    // classifier failure degrades the signal rather than losing it.
    _latestBehaviorScore = entry.classification?.behaviorScore ??
        (entry.result.status == AnomalyStatus.evaluated
            ? entry.result.aggregateScore
            : null);

    _emitDecision(
      evaluatedAt: entry.evaluatedAt,
      windowKey: entry.windowKey,
    );
  }

  void _emitDecision(
      {required DateTime evaluatedAt, required String windowKey}) {
    final fusionResult = _fusionEngine.fuse({
      'voice': _latestVoiceScore,
      'text': _latestTextScore,
      'behavior': _latestBehaviorScore,
    });

    // FR-4.2: apply the sustained-danger boost to the RAW fused score,
    // then classify THAT (not fusionResult's own pre-boost riskTier)
    // against thresholds that assume the post-boost 0-120 scale. See
    // SustainedDangerTracker's and RiskThresholds' doc comments.
    final boostedScore = _sustainedTracker.apply(
      fusionResult.finalScore,
      now: evaluatedAt,
    );
    final riskTier = classify(boostedScore, thresholds: _fusionEngine.thresholds);

    final decision = RiskDecision(
      riskTier: riskTier,
      fusionResult: fusionResult,
      boostedScore: boostedScore,
      evaluatedAt: evaluatedAt,
      windowKey: windowKey,
    );

    _latest = decision;
    if (!_controller.isClosed) {
      _controller.add(decision);
    }
  }
}
