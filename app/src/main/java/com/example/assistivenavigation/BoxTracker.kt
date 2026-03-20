package com.example.assistivenavigation

import kotlin.math.max
import kotlin.math.min

/**
 * Tracks a single bounding box across frames using a 4-state machine.
 *
 * Why this exists:
 *   YOLO drops detections on motion blur, lighting changes, and camera shake.
 *   A raw frame counter (confirmedFrames++) resets the whole pipeline on every
 *   miss. This class instead separates two very different situations:
 *     (a) A new object appearing  →  require N consecutive hits before we trust it
 *     (b) An existing track interrupted  →  predict where it went and wait for it
 *
 * States:
 *   IDLE        No object tracked. Waiting for first detection.
 *   CONFIRMING  Detection appeared but not yet trusted. One miss → back to IDLE.
 *   TRACKING    Confirmed track. Each frame we look for a matching detection.
 *   COASTING    Detection lost. We extrapolate position using last velocity and
 *               wait up to MAX_COAST_FRAMES before giving up.
 */
class BoxTracker {

    enum class State { IDLE, CONFIRMING, TRACKING, COASTING }

    companion object {
        /**
         * How many consecutive detections before we treat an object as real.
         * 4 frames at 30fps = ~133ms. Prevents reacting to single-frame ghosts.
         */
        private const val CONFIRM_FRAMES = 4

        /**
         * How many frames we hold a track without a detection before dropping it.
         * 20 frames ≈ 0.67 s at 30fps. Covers a typical camera shake or head turn.
         */
        private const val MAX_COAST_FRAMES = 20

        /**
         * Minimum IoU overlap to match a detection to the current track.
         * 0.2 is intentionally lower than the NMS threshold (0.5) because
         * the box may have moved during a coast period.
         */
        private const val MATCH_IOU_BASE = 0.2f

        /**
         * We reduce the IoU requirement by this amount per coast frame.
         * So after 5 missed frames the threshold drops from 0.20 → 0.12,
         * making it easier to re-associate after a longer gap.
         * Never drops below 0.05 to avoid matching completely wrong objects.
         */
        private const val MATCH_RELAX_PER_FRAME = 0.015f

        /**
         * Velocity smoothing factor (exponential moving average).
         * 0.25 means "25% new measurement, 75% old estimate" per frame.
         * Higher = more reactive to sudden motion, lower = smoother.
         */
        private const val VEL_ALPHA = 0.25f

        /**
         * Each coast frame, we expand the predicted box by this fraction.
         * This represents growing uncertainty — the farther we've predicted,
         * the larger the region the real box might be in.
         * Capped at 1.5× the original size.
         */
        private const val COAST_EXPAND_PER_FRAME = 0.02f
    }

    // ── Public state ──────────────────────────────────────────────────────────

    var state: State = State.IDLE
        private set

    /** True when the tracker is actively outputting a box. */
    val isActive: Boolean get() = state == State.TRACKING || state == State.COASTING

    /** True when we're running on predicted position (no live detection). */
    val isCoasting: Boolean get() = state == State.COASTING

    // ── Internal state ────────────────────────────────────────────────────────

    private var trackedBox:   BoundingBox? = null   // last confirmed detection
    private var predictedBox: BoundingBox? = null   // velocity-extrapolated box

    private var confirmedCount = 0
    private var coastCount     = 0

    // Box-centre velocity in normalised [0..1] units per frame
    private var velCx = 0f
    private var velCy = 0f

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Feed the latest YOLO detections for this frame.
     *
     * Returns the best estimate of the object's box if the track is active,
     * or null while confirming / after tracking is lost.
     */
    fun update(detections: List<BoundingBox>): BoundingBox? {
        val matched = findMatch(detections)

        when (state) {

            State.IDLE -> {
                // Take the highest-confidence detection as a candidate
                if (matched != null) {
                    trackedBox     = matched
                    confirmedCount = 1
                    velCx = 0f; velCy = 0f
                    state = State.CONFIRMING
                }
                return null   // not trusted yet
            }

            State.CONFIRMING -> {
                if (matched != null) {
                    updateVelocity(matched)
                    trackedBox = matched
                    confirmedCount++
                    if (confirmedCount >= CONFIRM_FRAMES) {
                        state = State.TRACKING
                    }
                } else {
                    // One miss during confirmation → restart.
                    // We're strict here so a ghost flash doesn't start audio.
                    reset()
                }
                return null   // still not trusted
            }

            State.TRACKING -> {
                return if (matched != null) {
                    updateVelocity(matched)
                    trackedBox   = matched
                    predictedBox = null
                    coastCount   = 0
                    trackedBox
                } else {
                    // First miss on a confirmed track → start coasting
                    coastCount   = 1
                    predictedBox = applyVelocity(trackedBox!!, coastCount)
                    state        = State.COASTING
                    predictedBox   // return predicted position immediately, no gap
                }
            }

            State.COASTING -> {
                return if (matched != null) {
                    // Re-acquired — snap back to a live detection
                    updateVelocity(matched)
                    trackedBox   = matched
                    predictedBox = null
                    coastCount   = 0
                    state        = State.TRACKING
                    trackedBox
                } else {
                    coastCount++
                    if (coastCount > MAX_COAST_FRAMES) {
                        reset()
                        null
                    } else {
                        // Keep extrapolating
                        predictedBox = applyVelocity(trackedBox!!, coastCount)
                        predictedBox
                    }
                }
            }
        }
    }

    /** Call when the session ends to clear all state. */
    fun reset() {
        state          = State.IDLE
        trackedBox     = null
        predictedBox   = null
        confirmedCount = 0
        coastCount     = 0
        velCx          = 0f
        velCy          = 0f
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Find the detection that best overlaps the current tracked/predicted box.
     *
     * When there's no reference yet (IDLE), just return the highest-confidence
     * detection because any IoU comparison would be meaningless.
     *
     * The IoU threshold relaxes slightly each coast frame so we can still
     * re-associate after the box has drifted.
     */
    private fun findMatch(detections: List<BoundingBox>): BoundingBox? {
        if (detections.isEmpty()) return null

        // No reference box yet → accept the strongest detection unconditionally
        val reference = predictedBox ?: trackedBox
        ?: return detections.firstOrNull()

        // Relax threshold the longer we've been coasting
        val threshold = (MATCH_IOU_BASE - coastCount * MATCH_RELAX_PER_FRAME)
            .coerceAtLeast(0.05f)

        var bestIou = threshold
        var bestBox: BoundingBox? = null

        for (det in detections) {
            val iou = calculateIoU(reference, det)
            if (iou > bestIou) {
                bestIou = iou
                bestBox = det
            }
        }

        return bestBox
    }

    /**
     * Update the stored velocity estimate using exponential smoothing.
     *
     * We measure displacement from whatever our last reference was
     * (predicted box if coasting, real box if tracking) to the new detection.
     * This lets velocity recovery work correctly after a coast period.
     */
    private fun updateVelocity(newBox: BoundingBox) {
        val ref = predictedBox ?: trackedBox ?: return
        val dCx = newBox.cx - ref.cx
        val dCy = newBox.cy - ref.cy
        velCx = VEL_ALPHA * dCx + (1f - VEL_ALPHA) * velCx
        velCy = VEL_ALPHA * dCy + (1f - VEL_ALPHA) * velCy
    }

    /**
     * Predict where the box will be after [coastFrames] frames of coasting.
     *
     * We apply accumulated velocity to the LAST REAL box (trackedBox), not
     * to the previous predicted box. This prevents drift from compounding —
     * each prediction is always anchored to the last confirmed position.
     *
     * The box is also slightly expanded to represent positional uncertainty.
     */
    private fun applyVelocity(box: BoundingBox, coastFrames: Int): BoundingBox {
        // Always extrapolate from the last real box, not the last prediction
        val newCx = (box.cx + velCx * coastFrames).coerceIn(0f, 1f)
        val newCy = (box.cy + velCy * coastFrames).coerceIn(0f, 1f)

        // Expand slightly to model growing uncertainty
        val expand = (1f + coastFrames * COAST_EXPAND_PER_FRAME).coerceAtMost(1.5f)
        val newW   = box.w * expand
        val newH   = box.h * expand

        val x1 = (newCx - newW / 2f).coerceIn(0f, 1f)
        val y1 = (newCy - newH / 2f).coerceIn(0f, 1f)
        val x2 = (newCx + newW / 2f).coerceIn(0f, 1f)
        val y2 = (newCy + newH / 2f).coerceIn(0f, 1f)

        return BoundingBox(
            x1 = x1, y1 = y1, x2 = x2, y2 = y2,
            cx = newCx, cy = newCy, w = newW, h = newH,
            cnf      = box.cnf,        // carry the last real confidence score
            cls      = box.cls,
            clsName  = box.clsName
        )
    }

    private fun calculateIoU(a: BoundingBox, b: BoundingBox): Float {
        val ix1   = max(a.x1, b.x1);  val iy1 = max(a.y1, b.y1)
        val ix2   = min(a.x2, b.x2);  val iy2 = min(a.y2, b.y2)
        val inter = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
        val union = a.w * a.h + b.w * b.h - inter
        return if (union > 0f) inter / union else 0f
    }
}