import 'dart:async';

import 'behavioral_anomaly_pipeline.dart';
import 'behavioral_monitoring_service.dart';
import 'emergency_response_orchestrator.dart';
import 'monitoring_service_controller.dart';
import 'voice_analysis_service.dart';

// NOTE: put this file at lib/services/safety_monitoring_controller.dart
// in the real project, next to the other services it imports. It also
// needs a reference to the app-level `emergencyOrchestrator` from
// main.dart — pass it in via `configure()` below, called once from
// main() right after the orchestrator is constructed.

/// The single, explicit on/off switch for ALL of NeuralSafe's
/// background safety monitoring: the foreground service (and its
/// persistent notification), continuous mic capture, behavioral
/// anomaly tracking, and the emergency-response orchestrator.
///
/// Transparency contract — do not weaken this without also updating
/// the dashboard UI:
///  - Monitoring is OFF on every app launch. main.dart only
///    initializes models/vaults; nothing here is started at launch.
///  - The only way to turn it on is the user flipping the "Active
///    Safety Monitoring" switch on the main screen themselves
///    (DashboardScreen._toggleMonitoring).
///  - Turning it on always shows Android's persistent "NeuralSafe —
///    Active Safety Monitoring" notification for as long as it runs.
///    There is no way to run the foreground service without it (by
///    Android design), and this app does not try to minimize, hide,
///    or mislabel that notification.
///
/// TODO(you): BehavioralMonitoringService and
/// EmergencyResponseOrchestrator weren't shared in the session that
/// wrote this file, so `.stop()` on each below is UNVERIFIED against
/// their real definitions — everything else in this codebase
/// (MonitoringServiceController, VoiceAnalysisService,
/// BehavioralAnomalyPipeline) follows a matched start()/stop() pair,
/// so it's likely already there or a one-line add (cancel whatever
/// timer/subscription start() set up). Send me those two files and
/// I'll verify/finish this.
class SafetyMonitoringController {
  SafetyMonitoringController._();

  static SafetyMonitoringController? _instance;
  static SafetyMonitoringController get instance =>
      _instance ??= SafetyMonitoringController._();

  final MonitoringServiceController _foregroundService =
      MonitoringServiceController();

  EmergencyResponseOrchestrator? _orchestrator;

  /// Call once from main(), right after `emergencyOrchestrator` is
  /// constructed, so this controller can start/stop it too. Until
  /// this is called, start() still works for everything except the
  /// orchestrator.
  void configure({required EmergencyResponseOrchestrator orchestrator}) {
    _orchestrator = orchestrator;
  }

  bool _isActive = false;
  bool get isActive => _isActive;

  final _statusController = StreamController<bool>.broadcast();
  Stream<bool> get statusChanges => _statusController.stream;

  /// Turns every monitoring component on, in the order that matters:
  /// the foreground service (and its persistent notification) must be
  /// up BEFORE continuous mic capture starts, per
  /// MonitoringServiceController's own doc comment. Returns false (and
  /// rolls everything back) if microphone permission is denied.
  Future<bool> start() async {
    if (_isActive) return true;

    await _foregroundService.start();

    final voiceStarted = await VoiceAnalysisService.instance.start();
    if (!voiceStarted) {
      await _foregroundService.stop();
      return false;
    }

    BehavioralMonitoringService.instance.start();
    BehavioralAnomalyPipeline.instance.start();
    _orchestrator?.start();

    _isActive = true;
    _statusController.add(true);
    return true;
  }

  Future<void> stop() async {
    if (!_isActive) return;

    _orchestrator?.stop();
    BehavioralAnomalyPipeline.instance.stop();
    BehavioralMonitoringService.instance.stop();
    await VoiceAnalysisService.instance.stop();
    await _foregroundService.stop();

    _isActive = false;
    _statusController.add(false);
  }
}
