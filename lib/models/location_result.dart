enum LocationStatus {
  success,
  permissionDenied,
  serviceDisabled,
  timeout,
  error
}

/// Result of a single location acquisition attempt. [latitude]/[longitude]
/// are only non-null when [status] is success.
class LocationResult {
  final double? latitude;
  final double? longitude;
  final DateTime timestamp;
  final LocationStatus status;

  /// True when [latitude]/[longitude] came from the device's cached
  /// last-known position rather than a freshly-acquired GPS fix — see
  /// LocationService's doc comment for why this is preferred as the
  /// FIRST attempt (it returns near-instantly, unlike a live fix which
  /// can take 10+ seconds indoors). Always false when [status] is not
  /// success. Callers that build a user-facing message (e.g.
  /// EmergencyAlertDispatcher) should surface this, since an
  /// approximate/stale location is still far more useful to a
  /// responder than none, but shouldn't be presented as if it were the
  /// victim's exact current position.
  final bool isApproximate;

  const LocationResult({
    required this.latitude,
    required this.longitude,
    required this.timestamp,
    required this.status,
    this.isApproximate = false,
  });

  bool get isSuccess => status == LocationStatus.success;

  /// A Google Maps link built from the coordinates, or null if this
  /// result did not succeed. Does not itself decide what to show in a
  /// message when unavailable — that's the caller's responsibility
  /// (see EmergencyAlertDispatcher's message template).
  String? get mapsLink {
    if (!isSuccess || latitude == null || longitude == null) return null;
    return 'https://maps.google.com/?q=$latitude,$longitude';
  }

  @override
  String toString() {
    return 'LocationResult(status: $status, latitude: $latitude, '
        'longitude: $longitude, isApproximate: $isApproximate, '
        'timestamp: $timestamp)';
  }
}
