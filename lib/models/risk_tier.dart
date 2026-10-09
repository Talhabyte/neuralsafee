/// Coarse risk classification derived from a fused danger score.
/// Naming mirrors Chapter 6.4's original NORMAL / MEDIUM RISK / HIGH
/// DANGER response tiers.
///
/// This enum and [classify] are pure classification only — no alert
/// dispatch, no SMS, no GPS, no UI. That remains out of scope until a
/// later, separately-approved phase.
enum RiskTier { normal, mediumRisk, highDanger }

/// Immutable score boundaries for [classify]. Both bounds are the
/// inclusive lower edge of their tier (e.g. a score exactly at
/// [mediumRiskMin] is mediumRisk, not normal).
///
/// SCALE NOTE: the proposal's original FR-4.3 specifies MEDIUM RISK
/// as 80-150 and HIGH DANGER as >150, against a weighted fusion
/// formula (FR-4.1) whose mathematical maximum is 100 and a sustained-
/// danger boost (FR-4.2) of +20 — i.e. a true achievable ceiling of
/// 120, not 150. The literal "150" threshold is unreachable under the
/// proposal's own stated formula; this was discovered during
/// implementation, not introduced by it. [highDangerMin] is
/// deliberately set here ABOVE the no-boost ceiling (100) and AT OR
/// BELOW the boosted ceiling (100 + SustainedDangerTracker's default
/// boostAmount of 20 = 120), so that — matching the proposal's actual
/// intent in NFR-S4 — a brief, un-sustained spike (capped at 100)
/// can never alone cross into highDanger; only a score that is both
/// high AND sustained (see SustainedDangerTracker) can. The bound
/// below is widened to [0, 120] accordingly (was [0, 100] before the
/// sustained-danger boost was implemented).
class RiskThresholds {
  final double mediumRiskMin;
  final double highDangerMin;

  const RiskThresholds({
    this.mediumRiskMin = 45.0,
    this.highDangerMin = 110.0,
  })  : assert(mediumRiskMin >= 0 && mediumRiskMin <= 120,
            'mediumRiskMin must be within [0, 120]'),
        assert(highDangerMin >= 0 && highDangerMin <= 120,
            'highDangerMin must be within [0, 120]'),
        assert(highDangerMin >= mediumRiskMin,
            'highDangerMin must be >= mediumRiskMin');

  @override
  String toString() =>
      'RiskThresholds(mediumRiskMin: $mediumRiskMin, highDangerMin: $highDangerMin)';
}

/// Pure classification: maps a score to a tier using [thresholds].
/// Deterministic, stateless — no side effects. [score] is expected to
/// already be the FINAL score — i.e. after SustainedDangerTracker's
/// boost has been applied, if any — not DangerScoreFusionEngine's raw
/// pre-boost finalScore.
RiskTier classify(double score,
    {RiskThresholds thresholds = const RiskThresholds()}) {
  if (score >= thresholds.highDangerMin) return RiskTier.highDanger;
  if (score >= thresholds.mediumRiskMin) return RiskTier.mediumRisk;
  return RiskTier.normal;
}
