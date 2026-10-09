import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

/// Thin wrapper over the native 'neuralsafe/monitoring_service'
/// MethodChannel. Starting this elevates the whole app process's OS
/// priority via an Android foreground service (MonitoringForegroundService.kt)
/// — a persistent, deliberately generic notification is required by
/// Android, it cannot run silently. The native service runs no Dart
/// code of its own and holds no FlutterEngine reference; the already-
/// running engine (Voice's mic stream, Behavioral's Timer) simply
/// survives backgrounding/screen-lock because the process itself is
/// protected while the service is active.
class MonitoringServiceController {
  static const _channel = MethodChannel('neuralsafe/monitoring_service');

  Future<void> start() async {
    // POST_NOTIFICATIONS is a runtime (not just manifest) permission on
    // Android 13+; without it the foreground service's required
    // notification may not display, though the process-priority
    // protection itself still applies regardless.
    await Permission.notification.request();
    await _channel.invokeMethod<void>('start');
  }

  Future<void> stop() async {
    await _channel.invokeMethod<void>('stop');
  }
}
