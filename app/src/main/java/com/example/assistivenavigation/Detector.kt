package com.example.assistivenavigation

import android.content.Context
import android.graphics.*
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*
import androidx.core.graphics.createBitmap

class Detector(context: Context) {

    companion object {

        private const val INPUT_SIZE = 320
        private const val NUM_DETECTIONS = 300

        private const val CONF_THRESHOLD = 0.25f
        private const val MAX_RESULTS = 10

        // temporal stabilization
        private const val STABILITY_FRAMES = 3
    }

    private val interpreter: Interpreter

    private val inputBuffer =
        ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())

    private val output =
        Array(1) { Array(NUM_DETECTIONS) { FloatArray(6) } }

    private var previousBest: Detection? = null
    private var stableCount = 0

    init {

        val model = loadModelFile(context)

        val options = Interpreter.Options().apply {

            setNumThreads(4)

            setUseNNAPI(true)

        }

        interpreter = Interpreter(model,options)
    }

    private fun loadModelFile(
        context: Context

    ): MappedByteBuffer {

        val fd = context.assets.openFd("yolov8n_float16.tflite")

        val input = FileInputStream(fd.fileDescriptor)

        val channel = input.channel

        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    fun detect(image: ImageProxy): List<Detection> {

        val bitmap = image.toBitmap()

        val letterboxed = letterbox(bitmap)

        bitmapToBuffer(letterboxed)

        interpreter.run(inputBuffer,output)

        val detections = parseDetections()

        return stabilize(detections)
    }

    // ---------------- LETTERBOX ----------------

    private fun letterbox(src: Bitmap): Bitmap {

        val scale = min(
            INPUT_SIZE.toFloat()/src.width,
            INPUT_SIZE.toFloat()/src.height
        )

        val matrix = Matrix()

        matrix.postScale(scale,scale)

        val resized = Bitmap.createBitmap(
            src,0,0,
            src.width,src.height,
            matrix,true
        )

        val padded = createBitmap(INPUT_SIZE, INPUT_SIZE)

        val canvas = Canvas(padded)

        val left = (INPUT_SIZE-resized.width)/2f
        val top = (INPUT_SIZE-resized.height)/2f

        canvas.drawBitmap(resized,left,top,null)

        return padded
    }

    // ---------------- INPUT BUFFER ----------------

    private fun bitmapToBuffer(bitmap: Bitmap){

        inputBuffer.rewind()

        val pixels = IntArray(INPUT_SIZE*INPUT_SIZE)

        bitmap.getPixels(
            pixels,0,INPUT_SIZE,
            0,0,INPUT_SIZE,INPUT_SIZE
        )

        for(pixel in pixels){

            val r = ((pixel shr 16) and 0xFF)/255f
            val g = ((pixel shr 8) and 0xFF)/255f
            val b = (pixel and 0xFF)/255f

            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
    }

    // ---------------- PARSE OUTPUT ----------------

    private fun parseDetections(): List<Detection> {

        val detections = mutableListOf<Detection>()

        for (i in 0 until NUM_DETECTIONS) {

            val score = output[0][i][4]

            if (score < CONF_THRESHOLD) continue

            val x1 = output[0][i][0]
            val y1 = output[0][i][1]

            val x2 = output[0][i][2]
            val y2 = output[0][i][3]

            val cls = output[0][i][5].toInt()

            val cx = (x1+x2)/2f


            val width = x2-x1
            val height = y2-y1

            val area = width*height

            val centerDist = abs(cx-0.5f)

            val centerWeight = 1f-centerDist

             val priority = score * (0.7f * area + 0.3f * centerWeight)

            detections.add(
                Detection(
                    x1.coerceIn(0f,1f),
                    y1.coerceIn(0f,1f),
                    x2.coerceIn(0f,1f),
                    y2.coerceIn(0f,1f),
                    score,
                    cls,
                    priority
                )
            )
        }

        return detections
            .sortedByDescending { it.priority }
            .take(MAX_RESULTS)
    }

    // ---------------- STABILIZER ----------------

    private fun stabilize(
        detections: List<Detection>
    ): List<Detection> {

        val best = detections.firstOrNull()

        if (best == null) {

            previousBest = null
            stableCount = 0

            return emptyList()
        }

        if (previousBest != null &&
            iou(best,previousBest!!) > 0.5f) {

            stableCount++

        } else {

            stableCount = 1
        }

        previousBest = best

        if (stableCount < STABILITY_FRAMES)
            return emptyList()

        return detections
    }

    private fun iou(a:Detection,b:Detection):Float{

        val x1 = max(a.x1,b.x1)
        val y1 = max(a.y1,b.y1)

        val x2 = min(a.x2,b.x2)
        val y2 = min(a.y2,b.y2)

        val inter = max(0f,x2-x1)*max(0f,y2-y1)

        val areaA = (a.x2-a.x1)*(a.y2-a.y1)
        val areaB = (b.x2-b.x1)*(b.y2-b.y1)

        return inter/(areaA+areaB-inter+1e-6f)
    }

    data class Detection(

        val x1:Float,
        val y1:Float,
        val x2:Float,
        val y2:Float,

        val score:Float,

        val cls:Int,

        val priority:Float
    )
}