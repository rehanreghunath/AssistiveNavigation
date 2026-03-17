package com.example.assistivenavigation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/**
 * Estimates the phone's movement speed using the linear accelerometer.
 *
 * TYPE_LINEAR_ACCELERATION already has gravity subtracted (by the OS sensor fusion),
 * so we don't need to worry about the 9.8 m/s^2 offset -> we get net motion only.
 *
 * Integration of a noisy signal drifts without bounds.
 * To fix this, we use:
 *   1. Velocity decay  -> each sample, velocity is multiplied by 0.95, so it
 *                        naturally bleeds to zero when there's no sustained push.
 *   2. Zero-velocity reset -> if acceleration magnitude stays below a threshold
 *                        for ZERO_FRAMES consecutive samples, velocity is zeroed.
 *                        This catches "the phone is sitting still on a desk" cleanly.
 */
class IMUProcessor(context: Context) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    // TYPE_LINEAR_ACCELERATION = gravity-compensated accelerometer
    private val linearAccel =
        sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    // ----- Internal state -----

    private val velocity = FloatArray(3)   // x, y, z in m/s
    private var lastTimestamp = 0L

    /** Small-motion threshold in m/s².  Below this we consider the phone still. */
    private val STILL_THRESHOLD = 0.06f

    /** How many consecutive still samples before we reset velocity. ~1 second at 50Hz */
    private val STILL_FRAMES = 50

    private var stillCount = 0

    /** Per-sample decay.  0.95 per ~20ms sample ≈ 7% per second bleed. */
    private val DECAY = 0.95f

    // ----- Public API -----

    /**
     * Estimated scalar speed in m/s.
     * @Volatile ensures reads from the camera thread see the latest value
     * written by the sensor thread without a full synchronisation barrier.
     */
    @Volatile
    var speed: Float = 0f
        private set

    fun start() {
        velocity.fill(0f)
        speed = 0f
        lastTimestamp = 0L
        stillCount = 0
        // SENSOR_DELAY_GAME ≈ 50 Hz.  Fast enough for frame-rate correlation
        // without hammering the battery.
        sensorManager.registerListener(this, linearAccel, SensorManager.SENSOR_DELAY_GAME)
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        velocity.fill(0f)
        speed = 0f
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_LINEAR_ACCELERATION) return

        // First sample -> just record the timestamp, no dt to integrate yet.
        if (lastTimestamp == 0L) {
            lastTimestamp = event.timestamp
            return
        }

        // dt in seconds between this sample and the previous one.
        // Timestamps are in nanoseconds from an arbitrary monotonic clock.
        val dt = (event.timestamp - lastTimestamp) / 1_000_000_000f
        lastTimestamp = event.timestamp

        // Sanity guard: ignore huge gaps (e.g., sensor sleeping then waking).
        if (dt <= 0f || dt > 0.5f) return

        val ax = event.values[0]
        val ay = event.values[1]
        val az = event.values[2]

        val accelMag = sqrt(ax * ax + ay * ay + az * az)

        if (accelMag < STILL_THRESHOLD) {
            stillCount++
            if (stillCount >= STILL_FRAMES) {
                // Phone has been still for ~1 second -> hard reset.
                velocity.fill(0f)
            }
            // Even when nearly still, apply decay so any residual drifts away.
            velocity[0] *= DECAY
            velocity[1] *= DECAY
            velocity[2] *= DECAY
        } else {
            stillCount = 0
            // Integrate:  v_new = v_old * decay  +  a * dt
            // The decay prevents unbounded growth on sustained acceleration.
            velocity[0] = velocity[0] * DECAY + ax * dt
            velocity[1] = velocity[1] * DECAY + ay * dt
            velocity[2] = velocity[2] * DECAY + az * dt
        }

        speed = sqrt(
            velocity[0] * velocity[0] +
                    velocity[1] * velocity[1] +
                    velocity[2] * velocity[2]
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* unused */ }
}