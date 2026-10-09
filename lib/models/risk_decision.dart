import 'risk_tier.dart';
import '../services/danger_score_fusion_engine.dart';

/// An immutable snapshot of a risk classification computed at one
/// point in time, from one behavioral evaluation window.  Pure state —
/// carries no side-effect behavior of its own.
///
/// [fusionResult] carries DangerScoreFusionEngine's RAW, pre-boost
/// output — its own `.finalScore` and `.riskTier` reflect the weighted
/// formula alone (FR-4.1), capped at 100, BEFORE the sustained-danger
/// boost (FR-4.2). It's kept here for transparency/debugging (e.g. to
/// see each module's contribution), but [fusionResult.riskTier] is
/// NOT the tier anything should act on.
///
/// [boostedScore] and [riskTier] on THIS class are the real, operative
/// values — [boostedScore] is [fusionResult.finalScore] with
/// SustainedDangerTracker's boost applied (if the sustained condition
/// was met), and [riskTier] is computed from THAT score. Every
/// consumer that decides whether to alert (EmergencyResponseOrchestrator
/// watching for a highDanger transition, the dashboard's display, etc.)
/// should read `riskTier`/`boostedScore` from here, not from
/// `fusionResult`.
class RiskDecision {
  final RiskTier riskTier;
  final FusionResult fusionResult;
  final double boostedScore;
  final DateTime evaluatedAt;
  final String windowKey;

  const RiskDecision({
    required this.riskTier,
    required this.fusionResult,
    required this.boostedScore,
    required this.evaluatedAt,
    required this.windowKey,
  });

  @override
  String toString() {
    return 'RiskDecision(riskTier: $riskTier, '
        'boostedScore: $boostedScore, '
        'rawScore: ${fusionResult.finalScore}, '
        'evaluatedAt: $evaluatedAt, windowKey: $windowKey)';
  }
}
