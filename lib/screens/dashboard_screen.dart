import 'dart:async';

import 'package:flutter/material.dart';

import '../services/emergency_contact_repository.dart';
import '../services/onboarding_repository.dart';
import '../services/risk_decision_engine.dart';
import '../services/safety_monitoring_controller.dart';
import '../services/text_analysis_service.dart';
import '../services/user_profile_repository.dart';
import '../services/voice_analysis_service.dart';
import '../services/voice_data_recorder.dart';
import 'setup_screen.dart';

/// NeuralSafe's real, transparent home screen — this is the app's
/// `home:` in main.dart, reached directly on launch. There is no
/// disguise, PIN, or hidden entry point: the app is always what it
/// says it is, and background monitoring is always visibly OFF until
/// the user turns it on themselves with the switch built into this
/// screen.
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final _onboardingRepo = OnboardingRepository();
  final _profileRepo = UserProfileRepository();
  final _contactRepo = EmergencyContactRepository();

  // The real, persistent, app-level fusion engine (constructed once in
  // main.dart). This screen only submits Text scores into it and reads
  // its latest decision back — it does NOT own its lifecycle.
  RiskDecisionEngine get _riskEngine => RiskDecisionEngine.instance;

  // --- Text module manual test hook (temporary, until real SMS
  // interception / MessageInterceptor is built). This is an explicit,
  // one-shot user action (type text, tap Analyze) — not continuous
  // background monitoring — so it isn't gated by the monitoring
  // switch below.
  final _textController = TextEditingController();
  TextAnalysisResult? _lastTextResult;
  bool _isAnalyzing = false;

  // --- Active Safety Monitoring: the single, explicit opt-in switch
  // that controls the foreground service, continuous mic capture,
  // behavioral tracking, and the emergency-response orchestrator, all
  // together, via SafetyMonitoringController. OFF by default; nothing
  // runs in the background until the user flips this themselves.
  bool _isMonitoringActive = false;
  bool _isTogglingMonitoring = false;

  // Voice result display only — score submission into the fusion
  // engine happens once, permanently, from main.dart, whenever Voice
  // is actually running.
  StreamSubscription<VoiceAnalysisResult>? _voiceSubscription;
  VoiceAnalysisResult? _lastVoiceResult;

  // --- Voice training-data recorder (development tool; see
  // VoiceDataRecorder). Records labeled clips through the same mic path
  // the live model uses, for retraining.
  VoiceClipLabel _clipLabel = VoiceClipLabel.normal;
  bool _isRecordingClip = false;
  String _clipStatus = '';
  Map<VoiceClipLabel, int> _clipCounts = const {};
  String _clipDirPath = '';

  @override
  void initState() {
    super.initState();
    _isMonitoringActive = SafetyMonitoringController.instance.isActive;
    _lastVoiceResult = VoiceAnalysisService.instance.latest;
    if (_isMonitoringActive) {
      _attachVoiceListener();
    }
    _refreshClipInfo();
  }

  Future<void> _refreshClipInfo() async {
    final counts = await VoiceDataRecorder.instance.clipCounts();
    final dir = await VoiceDataRecorder.instance.outputDirectory();
    if (!mounted) return;
    setState(() {
      _clipCounts = counts;
      _clipDirPath = dir.path;
    });
  }

  Future<void> _recordTrainingClip() async {
    setState(() {
      _isRecordingClip = true;
      _clipStatus = '';
    });
    try {
      await VoiceDataRecorder.instance.recordClip(
        label: _clipLabel,
        onStatus: (s) {
          if (mounted) setState(() => _clipStatus = s);
        },
      );
    } on StateError catch (e) {
      if (mounted) setState(() => _clipStatus = e.message);
    } catch (e) {
      if (mounted) setState(() => _clipStatus = 'Recording failed: $e');
    } finally {
      if (mounted) setState(() => _isRecordingClip = false);
      await _refreshClipInfo();
    }
  }

  void _attachVoiceListener() {
    _voiceSubscription?.cancel();
    _voiceSubscription = VoiceAnalysisService.instance.scores.listen((result) {
      setState(() => _lastVoiceResult = result);
    });
  }

  Future<void> _openSetup() async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const SetupScreen()),
    );
    // Refresh after returning, in case setup was just completed/edited.
    setState(() {});
  }

  Future<void> _analyzeText() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    setState(() => _isAnalyzing = true);
    try {
      final result = await TextAnalysisService.instance.analyze(text);
      _riskEngine.submitTextScore(result.score);
      setState(() {
        _lastTextResult = result;
        _isAnalyzing = false;
      });
    } catch (e) {
      setState(() => _isAnalyzing = false);
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Analysis failed: $e')));
      }
    }
  }

  /// The one place this screen turns real background monitoring on or
  /// off. Everything it does is visible to the user: the switch state
  /// itself, the status text explaining what's running, and (once on)
  /// Android's own persistent "NeuralSafe — Active Safety Monitoring"
  /// notification, which this app cannot hide even if it wanted to.
  Future<void> _toggleMonitoring() async {
    setState(() => _isTogglingMonitoring = true);

    if (_isMonitoringActive) {
      await SafetyMonitoringController.instance.stop();
      await _voiceSubscription?.cancel();
      _voiceSubscription = null;
      setState(() {
        _isMonitoringActive = false;
        _isTogglingMonitoring = false;
      });
      return;
    }

    final started = await SafetyMonitoringController.instance.start();
    setState(() => _isTogglingMonitoring = false);

    if (!started) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text(
            'Could not start monitoring — microphone permission is required.',
          ),
        ));
      }
      return;
    }

    _attachVoiceListener();
    setState(() => _isMonitoringActive = true);
  }

  @override
  void dispose() {
    _textController.dispose();
    _voiceSubscription?.cancel();
    // Monitoring itself is NOT stopped here — it's owned by
    // SafetyMonitoringController (app-level, foreground-service-
    // protected) so it keeps running if the user navigates away, not
    // just while this screen happens to be mounted. Only the display
    // subscription is torn down.
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final onboarding = _onboardingRepo.load();

    return Scaffold(
      appBar: AppBar(title: const Text('NeuralSafe')),
      body: Column(
        children: [
          _buildMonitoringToggleCard(onboarding.isComplete),
          const Divider(height: 1),
          Expanded(
            child: onboarding.isComplete ? _buildSummary() : _buildGetStarted(),
          ),
        ],
      ),
    );
  }

  Widget _buildMonitoringToggleCard(bool setupComplete) {
    final String statusText;
    if (_isMonitoringActive) {
      statusText = 'ON — listening for danger signals in your voice and '
          'phone activity. A persistent notification ("NeuralSafe — '
          'Active Safety Monitoring") stays visible the whole time, as '
          'Android requires.';
    } else if (setupComplete) {
      statusText = 'OFF — nothing runs in the background until you turn '
          'this on.';
    } else {
      statusText = 'Finish setup below to add an emergency contact before '
          'turning this on.';
    }

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                _isMonitoringActive ? Icons.shield : Icons.shield_outlined,
                color: _isMonitoringActive ? Colors.green : null,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Active Safety Monitoring',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(statusText,
                        style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              _isTogglingMonitoring
                  ? const SizedBox(
                      width: 24,
                      height: 24,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Switch(
                      value: _isMonitoringActive,
                      onChanged:
                          setupComplete ? (_) => _toggleMonitoring() : null,
                    ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildGetStarted() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Setup has not been completed yet.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _openSetup,
              child: const Text('Start Setup'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummary() {
    final profile = _profileRepo.load();
    final contact = _contactRepo.loadPrimary();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Profile', style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text('Name: ${profile?.name ?? '-'}'),
          Text('Preferred language: '
              '${profile?.preferredLanguage == 'ur' ? 'Urdu' : 'English'}'),
          const SizedBox(height: 24),
          const Text('Primary Emergency Contact',
              style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text('Name: ${contact?.name ?? '-'}'),
          Text('Phone: ${contact?.phoneNumber ?? '-'}'),
          const SizedBox(height: 24),
          OutlinedButton(
            onPressed: _openSetup,
            child: const Text('Edit Setup'),
          ),
          const Divider(height: 48),
          const Text('Text Module Test (temporary)',
              style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          TextField(
            controller: _textController,
            decoration: const InputDecoration(
              labelText: 'Message text',
              border: OutlineInputBorder(),
            ),
            maxLines: 3,
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: _isAnalyzing ? null : _analyzeText,
            child: Text(_isAnalyzing ? 'Analyzing...' : 'Analyze Text'),
          ),
          if (_lastTextResult != null) ...[
            const SizedBox(height: 16),
            Text('Predicted class: ${_lastTextResult!.predictedClass}'),
            Text('TextScore: ${_lastTextResult!.score.toStringAsFixed(2)}'),
            Text('Probabilities: ${_lastTextResult!.probabilities}'),
            const SizedBox(height: 8),
            Text(
              // Boosted score (post sustained-danger boost, FR-4.2) —
              // this is the score riskTier below is actually computed
              // from, not fusionResult's own raw pre-boost finalScore.
              'Fused Danger Score: '
              '${_riskEngine.latest?.boostedScore.toStringAsFixed(2) ?? '-'}',
            ),
            Text('Risk Tier: ${_riskEngine.latest?.riskTier ?? '-'}'),
          ],
          const Divider(height: 48),
          const Text('Voice Analysis (live)',
              style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(_isMonitoringActive
              ? 'Listening... (updates every 500ms)'
              : 'Turn on Active Safety Monitoring above to see live results.'),
          if (_lastVoiceResult != null) ...[
            const SizedBox(height: 16),
            Text('Classification: ${_lastVoiceResult!.classification}'),
            Text('VoiceScore: ${_lastVoiceResult!.score.toStringAsFixed(2)}'),
            Text('Probabilities: ${_lastVoiceResult!.probabilities}'),
            const SizedBox(height: 8),
            Text(
              // Boosted score (post sustained-danger boost, FR-4.2) —
              // this is the score riskTier below is actually computed
              // from, not fusionResult's own raw pre-boost finalScore.
              'Fused Danger Score: '
              '${_riskEngine.latest?.boostedScore.toStringAsFixed(2) ?? '-'}',
            ),
            Text('Risk Tier: ${_riskEngine.latest?.riskTier ?? '-'}'),
          ],
          const Divider(height: 48),
          const Text('Voice Data Recorder (temporary, for model training)',
              style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text(
            'Records 10-second labeled clips through the same microphone '
            'path the live model uses. Turn Active Safety Monitoring OFF '
            'first.\n'
            '• normal: talking normally, TV, room noise, silence\n'
            '• elevated: raised/angry voice, crying\n'
            '• danger: screaming, shouting for help, loud bangs',
          ),
          const SizedBox(height: 8),
          SegmentedButton<VoiceClipLabel>(
            segments: const [
              ButtonSegment(value: VoiceClipLabel.normal, label: Text('normal')),
              ButtonSegment(
                  value: VoiceClipLabel.elevated, label: Text('elevated')),
              ButtonSegment(value: VoiceClipLabel.danger, label: Text('danger')),
            ],
            selected: {_clipLabel},
            onSelectionChanged: _isRecordingClip
                ? null
                : (s) => setState(() => _clipLabel = s.first),
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: _isRecordingClip ? null : _recordTrainingClip,
            child: Text(_isRecordingClip
                ? 'Recording...'
                : 'Record 10 s clip (${_clipLabel.name})'),
          ),
          if (_clipStatus.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(_clipStatus),
          ],
          const SizedBox(height: 8),
          Text('Saved clips — normal: ${_clipCounts[VoiceClipLabel.normal] ?? 0}, '
              'elevated: ${_clipCounts[VoiceClipLabel.elevated] ?? 0}, '
              'danger: ${_clipCounts[VoiceClipLabel.danger] ?? 0}'),
          if (_clipDirPath.isNotEmpty)
            Text('Folder on phone: $_clipDirPath',
                style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
