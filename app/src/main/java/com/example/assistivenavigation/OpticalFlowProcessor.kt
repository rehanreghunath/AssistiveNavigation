package com.example.assistivenavigation

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import androidx.camera.core.ImageProxy
import kotlin.math.sqrt

/**
 * Computes sparse Lucas-Kanade optical flow on the camera's Y (luminance) plane.
 *
 * High-level flow per frame:
 *   1. extractGray()  — pull the Y plane from the ImageProxy into an OpenCV Mat,
 *                       then downsample 50% to cut processing time roughly in half.
 *   2. process()      — run Shi-Tomasi corner detection (when needed) and then
 *                       LK pyramid optical flow between the previous and current frame.
 *   3. Return FlowResult — a list of FlowPoints (each point has x,y, flow vector dx/dy,
 *                          magnitude, and which detection index it belongs to) plus a
 *                          per-detection average flow magnitude map.
 */
class OpticalFlowProcessor {

    companion object {
        /**
         * How many corners to track at most.
         * 150 is a good balance: enough spatial coverage without hammering the CPU.
         */
        private const val MAX_CORNERS = 150

        /**
         * Shi-Tomasi quality threshold (0..1).
         * Only corners with eigenvalue >= quality * best_eigenvalue are kept.
         * 0.01 is permissive -> picks up moderate corners, not just sharp ones.
         */
        private const val QUALITY_LEVEL = 0.01

        /**
         * Minimum pixel distance between two detected corners.
         * Prevents dense clusters in one region; 10px gives good spread.
         */
        private const val MIN_CORNER_DISTANCE = 10.0

        /**
         * LK search window half-size.  A 15x15 window means LK searches
         * in a 31x31 patch centred on each corner.  Larger => can track faster
         * motion but is more expensive.
         */
        private const val WIN_SIZE = 15

        /**
         * Pyramid levels for LK.
         * Level 0 = original scale, level 1 = half, level 2 = quarter, etc.
         * 3 levels means fast motion (many pixels per frame) still gets tracked.
         */
        private const val PYRAMID_LEVELS = 3

        /** Re-run corner detection every N frames, or when tracked count drops below MIN. */
        private const val RE_DETECT_EVERY = 20
        private const val MIN_TRACKED = 30

        /**
         * We process at half resolution (360p from 720p).
         * All returned coordinates are scaled back to the original image space.
         */
        private const val FLOW_SCALE = 0.5f
    }

    /**
     * One tracked point after optical flow.
     *
     * @param x            position in *original* image pixels (not downscaled)
     * @param y            position in *original* image pixels
     * @param dx           flow vector x -> how many pixels this point moved horizontally
     * @param dy           flow vector y -> vertical movement
     * @param magnitude    Euclidean length of (dx, dy) -> bigger = faster movement
     * @param detectionIndex  which entry in the detection list this point falls inside,
     *                        or -1 if it belongs to no bounding box
     */
    data class FlowPoint(
        val x: Float,
        val y: Float,
        val dx: Float,
        val dy: Float,
        val magnitude: Float,
        val detectionIndex: Int = -1
    )

    /**
     * Everything the caller needs from one processed frame.
     *
     * @param points                all successfully tracked FlowPoints
     * @param objectFlowMagnitudes  map of detectionIndex -> average flow magnitude
     *                              for points inside that bounding box
     * @param frameCount            ever-increasing frame counter (useful for timing)
     */
    data class FlowResult(
        val points: List<FlowPoint>,
        val objectFlowMagnitudes: Map<Int, Float>,
        val frameCount: Int
    )

    // ---- Internal state ----

    /** Grayscale Mat from the previous frame (at downscaled resolution). */
    private var prevGray: Mat? = null

    /** Corners tracked in the previous frame (at downscaled resolution). */
    private var prevPoints: MatOfPoint2f? = null

    private var frameCount = 0

    // -----------------------------------------------------------------
    //  Public API
    // -----------------------------------------------------------------

    /**
     * Extract the Y (luminance) plane from the camera ImageProxy and
     * return a downscaled grayscale Mat.
     *
     * Y plane is used because camera frames arrive in YUV_420_888 format.
     * The Y plane IS the grayscale image -> no color conversion needed.
     * This saves ~2–5ms per frame compared to converting a full ARGB bitmap.
     *
     * The caller MUST release the returned Mat when done:
     *     val gray = processor.extractGray(image)
     *     // ... use gray ...
     *     gray.release()
     */
    fun extractGray(image: ImageProxy): Mat {
        val plane = image.planes[0]          // planes[0] = Y (luminance) in YUV
        val buffer = plane.buffer
        buffer.rewind()                      // ensure we read from the beginning

        val rowStride   = plane.rowStride    // how many bytes between row starts
        val pixelStride = plane.pixelStride  // almost always 1 for the Y plane

        val w = image.width
        val h = image.height

        // Allocate a Mat for the full-resolution grayscale image
        val fullMat = Mat(h, w, CvType.CV_8UC1)
        val bytes = ByteArray(h * w)

        if (pixelStride == 1 && rowStride == w) {
            // Fast path: the Y plane is a flat, contiguous byte array.
            buffer.get(bytes)
        } else {
            // Slow path: the camera driver added row padding or channel interleave.
            // We copy pixel-by-pixel, skipping stride bytes.
            for (row in 0 until h) {
                for (col in 0 until w) {
                    bytes[row * w + col] = buffer[row * rowStride + col * pixelStride]
                }
            }
        }
        fullMat.put(0, 0, bytes)

        // Downsample to 50%.  INTER_AREA is the best interpolation for shrinking
        // (it averages a block of source pixels rather than just sampling one).
        val scaledW = (w * FLOW_SCALE).toInt()
        val scaledH = (h * FLOW_SCALE).toInt()
        val scaledMat = Mat()
        Imgproc.resize(fullMat, scaledMat, Size(scaledW.toDouble(), scaledH.toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA)

        fullMat.release()   // free the full-res copy immediately
        return scaledMat    // caller owns this -> must release it
    }

    /**
     * Step 2: Run optical flow.
     *
     * @param gray         the downscaled grayscale Mat from extractGray()
     * @param detections   the YOLO detections from the same frame
     * @param origW        original (not downscaled) image width, used to map
     *                     flow coordinates back to YOLO bounding box space
     * @param origH        original image height
     */
    fun process(
        gray: Mat,
        detections: List<Detector.Detection>,
        origW: Int,
        origH: Int
    ): FlowResult {
        frameCount++

        // ---- First frame ----
        // Nothing to compare against yet.  Store the frame and return empty.
        if (prevGray == null) {
            prevGray = gray.clone()
            prevPoints = detectCorners(gray)
            return FlowResult(emptyList(), emptyMap(), frameCount)
        }

        // ---- Re-detect corners ----
        // Done periodically so that corners which have drifted off-screen
        // or been occluded are replaced with fresh ones.
        val prevPts = prevPoints
        if (frameCount % RE_DETECT_EVERY == 0 ||
            prevPts == null || prevPts.rows() < MIN_TRACKED
        ) {
            prevPoints?.release()
            prevPoints = detectCorners(prevGray!!)
        }

        val pts = prevPoints
        if (pts == null || pts.rows() == 0) {
            // No corners to track -> store current frame and bail.
            updatePrevFrame(gray, null)
            return FlowResult(emptyList(), emptyMap(), frameCount)
        }

        // ---- Lucas-Kanade Pyramidal Optical Flow ----
        //
        //  Input:  prevGray (previous frame), gray (current frame),
        //          pts (corner positions in prevGray)
        //  Output: nextPts (where those corners are now in gray),
        //          status (1 = tracked successfully, 0 = lost),
        //          err (re-projection error per point)
        val nextPts = MatOfPoint2f()
        val status  = MatOfByte()
        val err     = MatOfFloat()

        Video.calcOpticalFlowPyrLK(
            prevGray!!, gray,
            pts, nextPts,
            status, err,
            Size(WIN_SIZE.toDouble(), WIN_SIZE.toDouble()),
            PYRAMID_LEVELS
        )

        val statusArr = status.toArray()
        val prevArr   = pts.toArray()
        val nextArr   = nextPts.toArray()

        val scaledW = gray.cols()
        val scaledH = gray.rows()

        // ---- Filter to only well-tracked points ----
        //
        // "Good" means: status == 1 (LK converged) AND the point hasn't
        // been pushed outside the image boundary.
        val goodPrev = mutableListOf<Point>()
        val goodNext = mutableListOf<Point>()

        for (i in statusArr.indices) {
            if (statusArr[i].toInt() != 1) continue
            val p = nextArr[i]
            if (p.x < 0 || p.x >= scaledW || p.y < 0 || p.y >= scaledH) continue
            goodPrev.add(prevArr[i])
            goodNext.add(nextArr[i])
        }

        // ---- Build FlowPoints ----
        //
        // Scale coordinates back to original image space so that the overlay
        // view can draw them correctly and YOLO box matching works.
        val invScale = 1f / FLOW_SCALE   // = 2.0 if FLOW_SCALE = 0.5

        val flowPoints = goodNext.mapIndexed { i, curr ->
            val prev = goodPrev[i]

            // Flow vector is in downscaled pixels; scale back to original pixels.
            val dx  = ((curr.x - prev.x) * invScale).toFloat()
            val dy  = ((curr.y - prev.y) * invScale).toFloat()
            val mag = sqrt(dx * dx + dy * dy)

            // Position in original image pixels
            val ox = (curr.x * invScale).toFloat()
            val oy = (curr.y * invScale).toFloat()

            // Normalise to [0, 1] to compare with YOLO boxes
            val normX = ox / origW
            val normY = oy / origH

            // Find which detection (if any) this point lives inside.
            // indexOfFirst returns -1 if no match.
            val detIdx = detections.indexOfFirst { d ->
                normX in d.x1..d.x2 && normY in d.y1..d.y2
            }

            FlowPoint(ox, oy, dx, dy, mag, detIdx)
        }

        // ---- Per-object average flow ----
        val objectFlows = mutableMapOf<Int, Float>()
        for (idx in detections.indices) {
            val group = flowPoints.filter { it.detectionIndex == idx }
            if (group.isNotEmpty()) {
                objectFlows[idx] = group.map { it.magnitude }.average().toFloat()
            }
        }

        // ---- Update state for next frame ----
        val nextCorners = if (goodNext.size >= MIN_TRACKED) {
            MatOfPoint2f(*goodNext.toTypedArray())
        } else {
            detectCorners(gray)   // too few survived -> fresh detect
        }
        updatePrevFrame(gray, nextCorners)

        // Release intermediate Mats to avoid native memory leaks
        nextPts.release()
        status.release()
        err.release()

        return FlowResult(flowPoints, objectFlows, frameCount)
    }

    /** Must be called when the screen is turned off or the user stops the session. */
    fun release() {
        prevGray?.release();  prevGray = null
        prevPoints?.release(); prevPoints = null
        frameCount = 0
    }

    // -----------------------------------------------------------------
    //  Private helpers
    // -----------------------------------------------------------------

    /**
     * Shi-Tomasi corner detection
     *
     * goodFeaturesToTrack scores each pixel by the smaller eigenvalue of the
     * gradient matrix in a local neighbourhood.  Large = distinctive corner.
     * It outputs a list of the best MAX_CORNERS pixels that are also at least
     * MIN_CORNER_DISTANCE apart (so we get spread across the whole frame).
     */
    private fun detectCorners(gray: Mat): MatOfPoint2f {
        val cornersI = MatOfPoint()
        Imgproc.goodFeaturesToTrack(
            gray, cornersI,
            MAX_CORNERS,
            QUALITY_LEVEL,
            MIN_CORNER_DISTANCE
        )
        val corners2f = MatOfPoint2f()
        if (!cornersI.empty()) {
            // MatOfPoint uses int coords; MatOfPoint2f uses float coords.
            // Both use org.opencv.core.Point (double) in the Java binding,
            // so a direct fromArray transfer is safe.
            corners2f.fromArray(*cornersI.toArray())
        }
        cornersI.release()
        return corners2f
    }

    /** Replace the stored previous frame + corner set. */
    private fun updatePrevFrame(gray: Mat, corners: MatOfPoint2f?) {
        prevGray?.release()
        prevGray = gray.clone()   // clone because the caller will release 'gray'
        prevPoints?.release()
        prevPoints = corners
    }
}