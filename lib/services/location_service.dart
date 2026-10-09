import 'dart:async';

import 'package:geolocator/geolocator.dart';

import '../models/location_result.dart';

/// Wraps geolocator for one-shot current-location acquisition. Every
/// geolocator call is behind an injectable function, so this is fully
/// unit testable without real GPS — same pattern as every other
/// native-adjacent service in this codebase (e.g. AppUsageService).
///
/// NFR-P3 requires an emergency SMS to be INITIATED within 5 seconds of
/// the Danger Score breaching HIGH DANGER. A live GPS fix can easily
/// take 10-15+ seconds indoors (confirmed on-device: a real test
/// returned `LocationStatus.timeout` after 15s indoors), and
/// EmergencyAlertDispatcher awaits [acquireCurrentLocation] BEFORE
/// sending the SMS — so blocking on a live fix risks silently violating
/// that 5-second budget in a real emergency.
///
/// To fix this, [acquireCurrentLocation] now tries the device's CACHED
/// last-known position first via [getLastKnownPosition] — this returns
/// near-instantly (no wait for satellites/network), which is exactly
/// what the 5-second budget needs. A last-known fix is marked
/// [LocationResult.isApproximate] so the alert message can be honest
/// about it being possibly stale, but a recent approximate location is
/// still far more useful to a responder than none. Only when there is
/// NO cached position at all (e.g. GPS has never gotten a fix on this
/// device since reboot/reinstall) does this fall back to waiting for a
/// live fix, now bounded by a shorter [timeout] than before, to keep
/// the worst case closer to the proposal's own 5-second number.
class LocationService {
  LocationService({
    Future<bool> Function()? isLocationServiceEnabled,
    Future<LocationPermission> Function()? checkPermission,
    Future<LocationPermission> Function()? requestPermission,
    Future<Position> Function()? getPosition,
    Future<Position?> Function()? getLastKnownPosition,
    DateTime Function()? clock,
    this.timeout = const Duration(seconds: 5),
  })  : _isLocationServiceEnabled =
            isLocationServiceEnabled ?? Geolocator.isLocationServiceEnabled,
        _checkPermission = checkPermission ?? Geolocator.checkPermission,
        _requestPermission = requestPermission ?? Geolocator.requestPermission,
        _getPosition = getPosition ??
            (() => Geolocator.getCurrentPosition(
                desiredAccuracy: LocationAccuracy.high)),
        _getLastKnownPosition =
            getLastKnownPosition ?? Geolocator.getLastKnownPosition,
        _clock = clock ?? DateTime.now;

  final Future<bool> Function() _isLocationServiceEnabled;
  final Future<LocationPermission> Function() _checkPermission;
  final Future<LocationPermission> Function() _requestPermission;
  final Future<Position> Function() _getPosition;
  final Future<Position?> Function() _getLastKnownPosition;
  final DateTime Function() _clock;

  /// Bound on the LIVE fix fallback only — the last-known lookup tried
  /// first has no timeout of its own, since it reads a cached value and
  /// returns near-instantly.
  final Duration timeout;

  Future<LocationResult> acquireCurrentLocation() async {
    final now = _clock();

    final serviceEnabled = await _isLocationServiceEnabled();
    if (!serviceEnabled) {
      return LocationResult(
        latitude: null,
        longitude: null,
        timestamp: now,
        status: LocationStatus.serviceDisabled,
      );
    }

    var permission = await _checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await _requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      return LocationResult(
        latitude: null,
        longitude: null,
        timestamp: now,
        status: LocationStatus.permissionDenied,
      );
    }

    // Attempt 1: cached last-known position — near-instant, no wait.
    // This is what lets a normal alert stay within NFR-P3's 5-second
    // budget even when a live fix would be slow.
    try {
      final lastKnown = await _getLastKnownPosition();
      if (lastKnown != null) {
        return LocationResult(
          latitude: lastKnown.latitude,
          longitude: lastKnown.longitude,
          timestamp: now,
          status: LocationStatus.success,
          isApproximate: true,
        );
      }
    } catch (_) {
      // Fall through to a live fix attempt below — a failure reading
      // the cache is not itself fatal.
    }

    // Attempt 2: live fix, only reached when there was no cached
    // position at all. Bounded by [timeout] (default 5s) rather than
    // the old 15s, so the worst case still roughly respects NFR-P3.
    try {
      final position = await _getPosition().timeout(timeout);
      return LocationResult(
        latitude: position.latitude,
        longitude: position.longitude,
        timestamp: now,
        status: LocationStatus.success,
      );
    } on TimeoutException {
      return LocationResult(
        latitude: null,
        longitude: null,
        timestamp: now,
        status: LocationStatus.timeout,
      );
    } catch (_) {
      return LocationResult(
        latitude: null,
        longitude: null,
        timestamp: now,
        status: LocationStatus.error,
      );
    }
  }
}
