package com.example.assistivenavigation

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    // Solid green → confirmed live detection
    private val boxPaintTracked = Paint().apply {
        color       = Color.GREEN
        strokeWidth = 6f
        style       = Paint.Style.STROKE
    }

    // Dashed orange → coasting (predicted position, no live detection)
    private val boxPaintCoasting = Paint().apply {
        color       = Color.rgb(255, 165, 0)
        strokeWidth = 6f
        style       = Paint.Style.STROKE
        pathEffect  = DashPathEffect(floatArrayOf(20f, 10f), 0f)
    }

    private val textPaint = Paint().apply {
        color    = Color.GREEN
        textSize = 42f
        style    = Paint.Style.FILL
    }

    private val flowPaintBackground = Paint().apply {
        color       = Color.argb(140, 0, 200, 255)
        style       = Paint.Style.FILL
        isAntiAlias = true
    }

    private val flowPaintObject = Paint().apply {
        color       = Color.argb(200, 255, 220, 30)
        style       = Paint.Style.FILL
        isAntiAlias = true
    }

    private var detections:  List<BoundingBox>                   = emptyList()
    private var isCoasting:  Boolean                              = false
    private var flowPoints:  List<OpticalFlowProcessor.FlowPoint> = emptyList()
    private var imageWidth   = 1
    private var imageHeight  = 1

    private val MAX_CIRCLE_RADIUS = 28f
    private val FLOW_SCALE_PIXELS = 20f
    private val MIN_CIRCLE_RADIUS = 3f

    /**
     * @param coasting pass boxTracker.isCoasting so the overlay can draw
     *                 an orange dashed box when predicting position.
     */
    fun updateDetections(results: List<BoundingBox>, coasting: Boolean = false) {
        detections = results
        isCoasting = coasting
        invalidate()
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
        drawFlowCircles(canvas)
        drawDetectionBoxes(canvas)
    }

    private fun drawFlowCircles(canvas: Canvas) {
        for (fp in flowPoints) {
            val vx     = fp.x / imageWidth  * width
            val vy     = fp.y / imageHeight * height
            val radius = (fp.magnitude / FLOW_SCALE_PIXELS * MAX_CIRCLE_RADIUS)
                .coerceIn(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS)
            val paint  = if (fp.detectionIndex >= 0) flowPaintObject else flowPaintBackground
            canvas.drawCircle(vx, vy, radius, paint)
        }
    }

    private fun drawDetectionBoxes(canvas: Canvas) {
        // Pick paint based on whether we're coasting or tracking
        val paint = if (isCoasting) boxPaintCoasting else boxPaintTracked

        for (det in detections) {
            val left   = det.x1 * width
            val top    = det.y1 * height
            val right  = det.x2 * width
            val bottom = det.y2 * height
            canvas.drawRect(left, top, right, bottom, paint)
            canvas.drawText(
                if (isCoasting) "~${det.clsName}" else det.clsName,
                left + 4f,
                top - 12f,
                textPaint
            )
        }
    }
}