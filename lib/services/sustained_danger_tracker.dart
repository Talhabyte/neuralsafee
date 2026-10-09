/// Tracks whether the fused Danger Score has remained continuously at
/// or above [boostThreshold] for at least [sustainedDuration], and if
/// so, adds [boostAmount] to the score — the sustained-danger boost
/// NeuralSafe's approved proposal specifies in FR-4.2 ("+20 point
/// boost applied if the Danger Score remains above 60 for five or
/// more consecutive seconds").
///
/// Why this exists: DangerScoreFusionEngine's weighted formula
/// (0.45×Voice + 0.35×Text + 0.20×Behavior, each input capped at 100)
/// mathematically tops out at 100 — it can never produce a score of
/// 110 or higher on its own, however extreme the inputs. This class is
/// what lets a SUSTAINED high score cross into the 100-120 range,
/// which is exactly what RiskThresholds.highDangerMin is now set
/// above the no-boost ceiling to require (see risk_tier.dart) — so a
/// single brief spike (max 100, un-boosted) cannot trigger HIGH
/// DANGER on its own, only a sustained one can. This directly
/// implements NFR-S4: "the sustained-danger boost mechanism must be
/// implemented to prevent isolated single-event spikes from
/// triggering HIGH DANGER responses."
///
/// Deliberately a separate, explicitly stateful class rather than
/// folded into DangerScoreFusionEngine, which is documented and
/// designed as pure/stateless (no continuity, no side effects) — the
/// "has this been sustained?" question needs state across fusion
/// cycles, so it lives here, one level up, next to
/// RiskDecisionEngine's own per-module state.
class SustainedDangerTracker {
  SustainedDangerTracker({
    this.boostThreshold = 60.0,
    this.sustainedDuration = const Duration(seconds: 5),
    this.boostAmount = 20.0,
    DateTime Function()? clock,
  }) : _clock = clock ?? DateTime.now;

  final double boostThreshold;
  final Duration sustainedDuration;
  final double boostAmount;
  final DateTime Function() _clock;

  /// When the score most recently, continuously, started being at or
  /// above [boostThreshold]. Null whenever the score is currently
  /// below the threshold.
  DateTime? _aboveSinceContinuous;

  /// Call once per fusion cycle with the RAW (pre-boost) fused score
  /// — i.e. `FusionResult.finalScore` from `DangerScoreFusionEngine`.
  /// Returns that score unchanged, or with [boostAmount] added once
  /// the sustained condition has actually been met.
  ///
  /// Resets the internal continuity timer the moment [rawScore] drops
  /// back below [boostThreshold], so a brief spike followed by a drop
  /// does not leave this "primed" to boost again without a fresh
  /// sustained period — matching the proposal's "remains above 60 for
  /// five or more CONSECUTIVE seconds" wording.
  double apply(double rawScore, {DateTime? now}) {
    final currentTime = now ?? _clock();

    if (rawScore < boostThreshold) {
      _aboveSinceContinuous = null;
      return rawScore;
    }

    _aboveSinceContinuous ??= currentTime;

    final sustainedFor = currentTime.difference(_aboveSinceContinuous!);
    if (sustainedFor >= sustainedDuration) {
      return rawScore + boostAmount;
    }
    return rawScore;
  }

  /// Clears tracked continuity state entirely. Call this whenever
  /// monitoring stops and later restarts, so a sustained period from a
  /// previous session can never silently carry over and immediately
  /// boost the very first score of a new one.
  void reset() {
    _aboveSinceContinuous = null;
  }
}
