package com.example.assistivenavigation

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) :
    View(context, attrs) {

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

    private var detections: List<Detector.Detection> =
        emptyList()

    fun updateDetections(results: List<Detector.Detection>) {
        detections = results
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {

        super.onDraw(canvas)

        for (det in detections) {

            val left = det.x1 * width
            val top = det.y1 * height
            val right = det.x2 * width
            val bottom = det.y2 * height

            canvas.drawRect(
                left,
                top,
                right,
                bottom,
                boxPaint
            )

            canvas.drawText(
                "Score: %.2f".format(det.score),
                left,
                top - 20,
                textPaint
            )
        }
    }
}