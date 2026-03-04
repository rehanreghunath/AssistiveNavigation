package com.example.assistivenavigation

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

class Detector(context: Context) {

    companion object {

        private const val INPUT_SIZE = 320
        private const val NUM_ATTR = 84
        private const val NUM_ANCHORS = 2100

        private const val OBJECTNESS_THRESHOLD = 0.35f
        private const val CONF_THRESHOLD = 0.4f
        private const val IOU_THRESHOLD = 0.45f

        private const val MAX_DETECTIONS = 5

        private const val TEMPORAL_CONFIRM_FRAMES = 3
    }

    private val interpreter: Interpreter

    private val inputBuffer =
        ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())

    private val output =
        Array(1) { Array(NUM_ATTR) { FloatArray(NUM_ANCHORS) } }

    // temporal smoothing
    private var previousBest: Detection? = null
    private var stableFrames = 0

    init {

        val model = loadModelFile(context, "yolov8n_int8.tflite")

        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
            setAllowFp16PrecisionForFp32(true)
        }

        interpreter = Interpreter(model, options)
    }

    private fun loadModelFile(
        context: Context,
        name: String
    ): MappedByteBuffer {

        val fd = context.assets.openFd(name)

        val input = FileInputStream(fd.fileDescriptor)

        val channel = input.channel

        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    fun detect(image: ImageProxy): List<Detection> {

        val start = System.currentTimeMillis()

        preprocess(image)

        inputBuffer.rewind()

        interpreter.run(inputBuffer, output)

        val detections = postprocess()

        val time = System.currentTimeMillis() - start

        Log.d("YOLO_TIMING","Inference ${time}ms")

        return detections
    }

    private fun sigmoid(x:Float):Float =
        1f/(1f+exp(-x))

    // ---------------- PREPROCESS ----------------

    private fun preprocess(image:ImageProxy) {

        inputBuffer.rewind()

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val width = image.width
        val height = image.height

        for (y in 0 until INPUT_SIZE) {

            val srcY = y * height / INPUT_SIZE

            for (x in 0 until INPUT_SIZE) {

                val srcX = x * width / INPUT_SIZE

                val yIndex = srcY * yRowStride + srcX

                val uvIndex =
                    (srcY/2)*uvRowStride +
                            (srcX/2)*uvPixelStride

                val Y = yBuf.get(yIndex).toInt() and 0xFF
                val U = uBuf.get(uvIndex).toInt() and 0xFF
                val V = vBuf.get(uvIndex).toInt() and 0xFF

                val r = Y + 1.402f*(V-128)
                val g = Y - 0.344f*(U-128) - 0.714f*(V-128)
                val b = Y + 1.772f*(U-128)

                inputBuffer.putFloat(r/255f)
                inputBuffer.putFloat(g/255f)
                inputBuffer.putFloat(b/255f)
            }
        }
    }

    // ---------------- POSTPROCESS ----------------

    private fun postprocess():List<Detection> {

        val rawDetections = ArrayList<Detection>()

        for (a in 0 until NUM_ANCHORS) {

            val objectness =
                sigmoid(output[0][4][a])

            if (objectness < OBJECTNESS_THRESHOLD)
                continue

            var bestClass = -1
            var bestScore = 0f

            for (c in 5 until NUM_ATTR) {

                val score =
                    sigmoid(output[0][c][a])

                if (score > bestScore) {
                    bestScore = score
                    bestClass = c-5
                }
            }

            val confidence = objectness*bestScore

            if (confidence < CONF_THRESHOLD)
                continue

            // decode box (pixel → normalized)

            val cx =
                output[0][0][a] / INPUT_SIZE

            val cy =
                output[0][1][a] / INPUT_SIZE

            val w =
                output[0][2][a] / INPUT_SIZE

            val h =
                output[0][3][a] / INPUT_SIZE

            val x1 = (cx - w/2).coerceIn(0f,1f)
            val y1 = (cy - h/2).coerceIn(0f,1f)

            val x2 = (cx + w/2).coerceIn(0f,1f)
            val y2 = (cy + h/2).coerceIn(0f,1f)

            rawDetections.add(
                Detection(
                    x1,y1,x2,y2,
                    confidence,
                    bestClass
                )
            )
        }

        val filtered = nms(rawDetections)

        val prioritized =
            filtered.sortedByDescending {

                val width = it.x2-it.x1
                val height = it.y2-it.y1
                val size = width*height

                val center =
                    abs((it.x1+it.x2)/2f - 0.5f)

                size * (1f-center)
            }

        val best = prioritized.firstOrNull()

        if (best != null) {

            if (previousBest != null &&
                iou(best,previousBest!!) > 0.5f) {

                stableFrames++

            } else {

                stableFrames = 1
            }

            previousBest = best
        }

        if (stableFrames < TEMPORAL_CONFIRM_FRAMES)
            return emptyList()

        return prioritized.take(MAX_DETECTIONS)
    }

    // ---------------- NMS ----------------

    private fun nms(
        detections:List<Detection>
    ):List<Detection>{

        val sorted =
            detections.sortedByDescending { it.score }

        val result = ArrayList<Detection>()

        val active = BooleanArray(sorted.size){true}

        for (i in sorted.indices) {

            if (!active[i]) continue

            val a = sorted[i]

            result.add(a)

            for (j in i+1 until sorted.size) {

                if (!active[j]) continue

                val b = sorted[j]

                if (iou(a,b) > IOU_THRESHOLD)
                    active[j] = false
            }
        }

        return result
    }

    private fun iou(a:Detection,b:Detection):Float{

        val x1 = max(a.x1,b.x1)
        val y1 = max(a.y1,b.y1)

        val x2 = min(a.x2,b.x2)
        val y2 = min(a.y2,b.y2)

        val inter =
            max(0f,x2-x1)*max(0f,y2-y1)

        val areaA =
            (a.x2-a.x1)*(a.y2-a.y1)

        val areaB =
            (b.x2-b.x1)*(b.y2-b.y1)

        return inter/(areaA+areaB-inter+1e-6f)
    }

    data class Detection(
        val x1:Float,
        val y1:Float,
        val x2:Float,
        val y2:Float,
        val score:Float,
        val cls:Int
    )
}