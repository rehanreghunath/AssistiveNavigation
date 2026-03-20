package com.example.assistivenavigation

import kotlin.math.max

/**
 * Estimates the distance to the primary detected object in metres.
 *
 * Combines three sources of evidence, weighted by how trustworthy each
 * is at this moment:
 *
 *  (A) Bounding-box width   -> always available; bigger box = closer object
 *  (B) Optical-flow ratio   -> how fast the object moves vs the background
 *  (C) Physics estimate     -> D = K * imuSpeed / objectFlow
 *
 * The three are blended so (C) dominates while walking and (A) takes over
 * when the phone is stationary.
 */
class DistanceEstimator {

    companion object {
        /**
         * Focal constant K ≈ focal_length_px / fps.
         * For a 720p phone camera (~840 px focal length) at 30 fps: K ≈ 28.
         * Tune this if distances feel consistently off:
         *   too far -> lower K, too close -> raise K.
         */
        private const val FOCAL_K = 28f

        private const val MIN_SPEED = 0.08f   // m/s; below this, physics is unreliable
        private const val MIN_FLOW  = 1.0f    // px/frame; below this, flow is noise
        private const val MIN_BG_FLOW = 1.5f  // px/frame; minimum to trust ratio method

        private const val MIN_DIST  = 0.4f    // metres
        private const val MAX_DIST  = 6.0f    // metres

        /**
         * Exponential smoothing factor. 0.88 means the output moves slowly
         * toward new measurements -> about 8 frames to close half the gap.
         * Increase (-> 0.95) for a more stable sound; decrease (-> 0.7) for
         * faster tracking.
         */
        private const val ALPHA = 0.88f
    }

    private var smoothed    = 2f
    private var initialized = false

    /**
     * Call once per confirmed detection frame.
     *
     * @param objectFlowMag      average flow magnitude of points inside the
     *                           detection bounding box, in original-image px/frame
     * @param backgroundFlowMag  average flow of ALL other points (ego-motion),
     *                           in original-image px/frame
     * @param imuSpeed           device speed from IMUProcessor, in m/s
     * @param bboxWidth          normalised bounding-box width in [0..1]
     */
    fun estimate(
        objectFlowMag: Float,
        backgroundFlowMag: Float,
        imuSpeed: Float,
        bboxWidth: Float
    ): Float {

        // ---- (A) Bounding-box baseline ----
        // If the box spans ~50% of image width, we assume ~1 m.
        // The formula "1 / (width * 2)" gives roughly that: width=0.5 → 1 m.
        val bboxDist = (1f / max(bboxWidth * 2f, 0.05f))
            .coerceIn(MIN_DIST, MAX_DIST)

        // ---- (B) Flow-ratio refinement ----
        // If background is moving faster than the object → object is farther.
        // ratio > 1 → object is slower than background → push distance up.
        // ratio < 1 → object is faster than background → pull distance down.
        val ratioDist: Float? = if (backgroundFlowMag > MIN_BG_FLOW) {
            val ratio = (backgroundFlowMag / max(objectFlowMag, 0.1f))
                .coerceIn(0.2f, 5f)
            (bboxDist * ratio).coerceIn(MIN_DIST, MAX_DIST)
        } else null

        // ---- (C) Physics estimate ----
        // D = K * V / F  (derived from the optical-flow equation)
        val physicsAvailable = imuSpeed > MIN_SPEED && objectFlowMag > MIN_FLOW
        val physicsDist: Float? = if (physicsAvailable) {
            (FOCAL_K * imuSpeed / objectFlowMag).coerceIn(MIN_DIST, MAX_DIST)
        } else null

        // ---- Blend weights ----
        // speedWeight rises from 0 → 0.5 as the user walks faster
        val speedWeight = if (physicsAvailable)
            ((imuSpeed - MIN_SPEED) / 0.4f).coerceIn(0f, 0.5f) else 0f

        // ratioWeight is active whenever background is moving, but capped so
        // it doesn't take over completely
        val ratioWeight = if (ratioDist != null)
            0.3f * (1f - speedWeight) else 0f

        val bboxWeight = 1f - speedWeight - ratioWeight

        val raw = (physicsDist ?: 0f) * speedWeight +
                (ratioDist  ?: bboxDist) * ratioWeight +
                bboxDist * bboxWeight

        // ---- Exponential smoothing ----
        smoothed = if (!initialized) { initialized = true; raw }
        else ALPHA * smoothed + (1f - ALPHA) * raw

        return smoothed
    }

    /** Call when the session stops so the next session starts fresh. */
    fun reset() {
        smoothed    = 2f
        initialized = false
    }
}