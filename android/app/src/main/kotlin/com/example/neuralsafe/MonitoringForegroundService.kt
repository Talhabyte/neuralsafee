package com.example.neuralsafe

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.pm.ServiceInfo
import android.content.Intent
import android.os.Build
import android.os.IBinder

/**
 * A minimal Android foreground service whose only job is to elevate
 * this process's OS priority so it survives being backgrounded or the
 * screen being locked. It runs NO Dart code of its own and holds no
 * FlutterEngine reference — calling startForeground() is the entire
 * mechanism; the already-running Flutter engine (Voice's continuous
 * mic stream, BehavioralAnomalyPipeline's Timer, etc.) simply keeps
 * running because the process itself is protected.
 *
 * Started/stopped from Dart via the 'neuralsafe/monitoring_service'
 * MethodChannel, which is itself only ever invoked from
 * SafetyMonitoringController — i.e. only after the user explicitly
 * flips the "Active Safety Monitoring" toggle on the main screen.
 *
 * The notification is intentionally clear and honest about what is
 * running ("NeuralSafe" / "Active Safety Monitoring"), never generic
 * or disguised as something else. Android requires a foreground
 * service to show SOME persistent, ongoing notification — it cannot
 * run silently — and this app does not try to work around that.
 */
class MonitoringForegroundService : Service() {

    companion object {
        private const val CHANNEL_ID = "neuralsafe_monitoring_channel"
        private const val NOTIFICATION_ID = 1001
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        return START_STICKY
    }

    private fun buildNotification(): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            val channel = NotificationChannel(
                CHANNEL_ID,
                "NeuralSafe Safety Monitoring",
                // IMPORTANCE_LOW (not MIN): the notification should be
                // visibly present in the shade, not hidden/minimized -
                // transparency means the user can actually see it.
                NotificationManager.IMPORTANCE_LOW,
            )
            manager.createNotificationChannel(channel)
        }

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("NeuralSafe")
            .setContentText("Active Safety Monitoring")
            .setSmallIcon(applicationInfo.icon)
            .setOngoing(true)
            .build()
    }
}