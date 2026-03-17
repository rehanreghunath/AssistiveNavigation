package com.example.assistivenavigation

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    // ---- Detection boxes ----
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 6f
        style = Paint.Style.STROKE
    }
    private val textPaint = Paint().apply {
        color = Color.GREEN
        textSize = 48f
        style = Paint.Style.FILL
    }

    // ---- Optical flow circles ----

    /**
     * Points NOT inside any detection box
     */
    private val flowPaintBackground = Paint().apply {
        color = Color.argb(140, 0, 200, 255)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    /**
     * Points inside a detection bounding box
     */
    private val flowPaintObject = Paint().apply {
        color = Color.argb(200, 255, 220, 30)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // ---- State ----
    private var detections: List<Detector.Detection> = emptyList()
    private var flowPoints: List<OpticalFlowProcessor.FlowPoint> = emptyList()

    /**
     * The image dimensions that the flow coordinates are in.
     * We need these to map from image pixels -> view pixels.
     */
    private var imageWidth  = 1
    private var imageHeight = 1

    /**
     * A flow magnitude of FLOW_SCALE_PIXELS corresponds to MAX_CIRCLE_RADIUS pixels
     * on screen.  Tune FLOW_SCALE_PIXELS based on what "fast" looks like in practice
     * (typically 15–30 px of flow per frame when walking quickly).
     */
    private val MAX_CIRCLE_RADIUS  = 28f
    private val FLOW_SCALE_PIXELS  = 20f
    private val MIN_CIRCLE_RADIUS  = 3f    // always at least a visible dot

    fun updateDetections(results: List<Detector.Detection>) {
        detections = results
        invalidate()   // Redraw the view
    }

    fun updateFlow(
        points: List<OpticalFlowProcessor.FlowPoint>,
        imgWidth: Int,
        imgHeight: Int
    ) {
        flowPoints  = points
        imageWidth  = imgWidth
        imageHeight = imgHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Draw flow circles FIRST so detection boxes appear on top.
        drawFlowCircles(canvas)
        drawDetectionBoxes(canvas)
    }

    // -----------------------------------------------------------------

    private fun drawFlowCircles(canvas: Canvas) {
        for (fp in flowPoints) {
            // Map from image-space coordinates to view-space coordinates.
            // The view might be a different size than the image (e.g., 1080×1920 view
            // but 1280×720 image), so we normalise through fractions.
            val vx = fp.x / imageWidth  * width
            val vy = fp.y / imageHeight * height

            // Radius is proportional to flow magnitude, clamped to a readable range.
            val radius = (fp.magnitude / FLOW_SCALE_PIXELS * MAX_CIRCLE_RADIUS)
                .coerceIn(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS)

            val paint = if (fp.detectionIndex >= 0) flowPaintObject else flowPaintBackground
            canvas.drawCircle(vx, vy, radius, paint)
        }
    }

    private fun drawDetectionBoxes(canvas: Canvas) {
        for (det in detections) {
            val left   = det.x1 * width
            val top    = det.y1 * height
            val right  = det.x2 * width
            val bottom = det.y2 * height

            canvas.drawRect(left, top, right, bottom, boxPaint)
            canvas.drawText(
                "%.2f".format(det.score),
                left + 4f,
                top - 12f,
                textPaint
            )
        }
    }
}