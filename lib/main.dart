import 'package:flutter/material.dart';
import 'screens/calculator_screen.dart';
import 'services/emergency_alert_dispatcher.dart';
import 'services/emergency_response_orchestrator.dart';
import 'services/feature_vault_service.dart';
import 'services/location_service.dart';
import 'services/risk_decision_engine.dart';
import 'services/safety_monitoring_controller.dart';
import 'services/secure_vault_service.dart';
import 'services/sms_dispatch_service.dart';
import 'services/alert_cooldown_manager.dart';
import 'services/text_analysis_service.dart';
import 'services/voice_analysis_service.dart';

/// IMPORTANT: constructed with dryRun: true below — real SMS sending is
/// disabled. Flip dryRun to false only deliberately, once the pipeline
/// is verified on a physical device with a real test contact you
/// control (see EmergencyAlertDispatcher's own doc comment).
late final EmergencyResponseOrchestrator emergencyOrchestrator;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SecureVaultService.instance.init();
  debugPrint('[NeuralSafe] Secure vault initialized: '
      '${SecureVaultService.instance.isInitialized}');

  // Separate encrypted box from SecureVaultService's — holds
  // BehavioralBaseline and AnomalyResultLogEntry data (Phase 4).
  await FeatureVaultService.instance.init();
  debugPrint('[NeuralSafe] Feature vault initialized: '
      '${FeatureVaultService.instance.isInitialized}');

  // ---------------------------------------------------------------
  // Everything from here down is MODEL/SERVICE INITIALIZATION ONLY:
  // loading TFLite models, opening streams, wiring listeners. None of
  // it starts the microphone, behavioral tracking, or the foreground
  // service. Monitoring is OFF on every launch — the only thing that
  // turns it on is the user flipping "Active Safety Monitoring" on
  // the main screen, via SafetyMonitoringController (see that file).
  // ---------------------------------------------------------------

  await TextAnalysisService.instance.init();
  debugPrint('[NeuralSafe] Text analysis service initialized: '
      '${TextAnalysisService.instance.isInitialized}');
  await VoiceAnalysisService.instance.init();
  debugPrint('[NeuralSafe] Voice analysis service initialized: '
      '${VoiceAnalysisService.instance.isInitialized}');

  // The real, persistent fusion engine. Constructing/accessing it here
  // just wires its subscription to the (not-yet-running) behavioral
  // pipeline stream — harmless until that pipeline actually starts.
  final riskEngine = RiskDecisionEngine.instance;

  // Permanent wiring, not a start: this only makes sure that IF/WHEN
  // Voice is running, its scores reach the fusion engine. It sits
  // inert — the stream simply never emits — until
  // SafetyMonitoringController.start() actually starts the mic.
  VoiceAnalysisService.instance.scores.listen((result) {
    riskEngine.submitVoiceScore(result.score);
  });

  emergencyOrchestrator = EmergencyResponseOrchestrator(
    riskDecisions: riskEngine.decisions,
    dispatcher: EmergencyAlertDispatcher(
      locationService: LocationService(),
      smsDispatchService: SmsDispatchService(),
      cooldownManager: AlertCooldownManager(),
      // Confirmed working on-device 2026-10-04 with a real, dryRun:
      // false test to the user's own number — both the highDanger-
      // transition path and the decoy PIN's triggerManualAlert() path
      // deliver a real SMS with the correct message. Reverted back to
      // dryRun: true immediately after that confirmation, as planned.
      // Flip to false again only deliberately, with a real test
      // contact you control (see EmergencyAlertDispatcher's own doc
      // comment).
      dryRun: true,
    ),
  );
  SafetyMonitoringController.instance.configure(
    orchestrator: emergencyOrchestrator,
  );
  debugPrint('[NeuralSafe] Emergency response orchestrator constructed '
      '(dryRun: true — no real SMS will be sent). Not started: it only '
      'runs while Active Safety Monitoring is switched on.');

  runApp(const NeuralSafeApp());
}

class NeuralSafeApp extends StatelessWidget {
  const NeuralSafeApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeuralSafe',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(brightness: Brightness.dark, useMaterial3: true),
      home: const CalculatorScreen(),
    );
  }
}
