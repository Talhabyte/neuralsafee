import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../main.dart';
import '../services/calculator_engine.dart';
import '../services/secret_code_service.dart';
import 'dashboard_screen.dart';

class CalculatorScreen extends StatefulWidget {
  const CalculatorScreen({super.key});

  @override
  State<CalculatorScreen> createState() => _CalculatorScreenState();
}

class _CalculatorScreenState extends State<CalculatorScreen> {
  final _engine = CalculatorEngine();
  final _secretCodes = SecretCodeService();

  String _input = '';
  String _display = '0';

  static const _buttons = [
    'C', '⌫', '%', '/',
    '7', '8', '9', '*',
    '4', '5', '6', '-',
    '1', '2', '3', '+',
    '0', '.', '=',
  ];

  void _onPressed(String label) {
    switch (label) {
      case 'C':
        setState(() {
          _input = '';
          _display = '0';
        });
        return;
      case '⌫':
        setState(() {
          if (_input.isNotEmpty) {
            _input = _input.substring(0, _input.length - 1);
          }
          _display = _input.isEmpty ? '0' : _input;
        });
        return;
      case '=':
        _handleEquals();
        return;
      default:
        setState(() {
          _input += label;
          _display = _input;
        });
    }
  }

  void _handleEquals() {
    final fullInput = '$_input=';

    // Check silently, before doing anything visible, whether this
    // "calculation" is actually a hidden command. Nothing about the
    // UI reveals that this check is happening.
    final action = _secretCodes.evaluate(fullInput);

    switch (action) {
      case CodeAction.openDashboard:
        setState(() {
          _input = '';
          _display = '0';
        });
        Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => const DashboardScreen()),
        );
        return;
      case CodeAction.silentDecoyAlarm:
        // Resets the display like nothing happened — the whole point
        // of the decoy code — while a real (dry-run by default; see
        // main.dart's emergencyOrchestrator construction) alert
        // dispatch runs in the background through the exact same
        // EmergencyAlertDispatcher + AlertCooldownManager a genuine
        // highDanger transition would use. Fire-and-forget from the
        // UI's perspective: never awaited, so there is no delay and
        // no risk of this screen visibly hanging while location is
        // acquired or SMS is sent.
        setState(() {
          _input = '';
          _display = '0';
        });
        unawaited(
          emergencyOrchestrator.triggerManualAlert().then((result) {
            debugPrint(
              '[NeuralSafe] Decoy PIN triggered manual alert dispatch: '
              '${result.outcome}',
            );
          }),
        );
        return;
      case CodeAction.none:
        break;
    }

    // Not a hidden command — behave like a normal calculator.
    try {
      final result = _engine.evaluate(_input);
      setState(() {
        _display = _formatResult(result);
        _input = _display;
      });
    } catch (_) {
      setState(() {
        _display = 'Error';
        _input = '';
      });
    }
  }

  String _formatResult(double value) {
    if (value == value.roundToDouble()) {
      return value.toInt().toString();
    }
    return value.toString();
  }

  Color _colorFor(String label) {
    if (label == '=') return Colors.orange;
    if ('+-*/'.contains(label)) return Colors.orange.shade700;
    if (label == 'C' || label == '⌫' || label == '%') {
      return Colors.grey.shade600;
    }
    return Colors.grey.shade800;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(
              maxWidth: 450,
            ),
            child: Column(
              children: [
                // Calculator display
                SizedBox(
                  height: 120,
                  width: double.infinity,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 24,
                      vertical: 16,
                    ),
                    alignment: Alignment.bottomRight,
                    child: Text(
                      _display,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 56,
                        fontWeight: FontWeight.w300,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ),

                // Calculator buttons
                Expanded(
                  child: _buildButtonGrid(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildButtonGrid() {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate dynamic dimensions to ensure all 5 rows fit perfectly within available height
        const double spacing = 10.0;
        const double padding = 12.0;

        final double totalVerticalSpacing = (spacing * 4) + (padding * 2);
        final double itemHeight = (constraints.maxHeight - totalVerticalSpacing) / 5;

        final double totalHorizontalSpacing = (spacing * 3) + (padding * 2);
        final double itemWidth = (constraints.maxWidth - totalHorizontalSpacing) / 4;

        final double aspectRatio = itemWidth / itemHeight;

        return GridView.builder(
          padding: const EdgeInsets.all(padding),
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 4,
            childAspectRatio: aspectRatio > 0 ? aspectRatio : 1.0,
            crossAxisSpacing: spacing,
            mainAxisSpacing: spacing,
          ),
          itemCount: _buttons.length,
          itemBuilder: (context, index) {
            final label = _buttons[index];

            return ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: _colorFor(label),
                shape: const CircleBorder(),
                padding: EdgeInsets.zero,
              ),
              onPressed: () => _onPressed(label),
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 24,
                  color: Colors.white,
                ),
              ),
            );
          },
        );
      },
    );
  }
}
