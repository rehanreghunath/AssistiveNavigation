package com.example.assistivenavigation

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import com.example.assistivenavigation.Constants.MODEL_PATH
import com.example.assistivenavigation.Constants.LABELS_PATH
import kotlin.math.max
import kotlin.collections.emptyList

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var startButton: Button
    private lateinit var uploadButton: Button
    private lateinit var overlayView: OverlayView
    private lateinit var debugOverlay: TextView
    private lateinit var videoFrameView: android.widget.ImageView
    private var videoRunning = false
    private var videoTestJob: java.util.concurrent.Future<*>? = null

    private lateinit var detector: Detector
    private lateinit var audioEngine: AudioEngine

    private lateinit var opticalFlowProcessor: OpticalFlowProcessor
    private lateinit var imuProcessor: IMUProcessor

    private var smoothedAzimuth = 0f
    private var lastUIUpdate = 0L
    private var lastAudioUpdateTime = 0L

    private var systemRunning = false
    private var audioStarted = false

    private var confirmedFrames = 0
    private val REQUIRED_CONFIRM_FRAMES = 5

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val isProcessing = AtomicBoolean(false)

    private var cameraProvider: ProcessCameraProvider? = null

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else finish()
        }

    private val videoPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            uri ?: return@registerForActivityResult
            // Resolve to a file path the retriever can open
            val path = getRealPathFromUri(uri)
            if (path != null) startVideoTest(path)
            else {
                // Fallback: pass the Uri directly via FileDescriptor
                startVideoTestFromUri(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        OpenCVLoader.initLocal()
        setContentView(R.layout.activity_main)

        previewView   = findViewById(R.id.previewView)
        overlayView   = findViewById(R.id.overlayView)
        debugOverlay  = findViewById(R.id.debugOverlay)
        startButton   = findViewById(R.id.startButton)
        uploadButton  = findViewById(R.id.uploadButton)

        detector    = Detector(baseContext, MODEL_PATH, LABELS_PATH)
        audioEngine = AudioEngine()
        opticalFlowProcessor = OpticalFlowProcessor()
        imuProcessor = IMUProcessor(this)

        previewView.visibility = View.INVISIBLE
        videoFrameView = findViewById(R.id.videoFrameView)
        debugOverlay.setText(R.string.waiting)

        uploadButton.setOnClickListener {
            if(!videoRunning){
                videoPicker.launch("video/*")
            } else {
                stopVideoTest()
            }
        }

        startButton.setOnClickListener {
            if (!systemRunning) {
                systemRunning = true
                startButton.setText(R.string.stop)
                previewView.visibility = View.VISIBLE
                debugOverlay.text = ""
                imuProcessor.start()          // start sensor
                checkPermissionAndStart()
            } else {
                stopSystem()
            }
        }
    }

    private fun stopSystem() {
        systemRunning = false
        startButton.setText(R.string.start)

        cameraProvider?.unbindAll()
        previewView.visibility = View.INVISIBLE
        overlayView.updateDetections(emptyList())
        overlayView.updateFlow(emptyList(), 1, 1)   // clear flow overlay
        debugOverlay.setText(R.string.waiting)
        confirmedFrames = 0

        if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }

        imuProcessor.stop()                    // stop sensor
        opticalFlowProcessor.release()         // free OpenCV Mats
    }

    private fun checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera()
        else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy(
                    AspectRatio.RATIO_16_9,
                    AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .setResolutionStrategy(ResolutionStrategy(
                    Size(1280, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { image -> analyzeFrame(image) }

            cameraProvider?.unbindAll()
            cameraProvider?.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (!systemRunning) { image.close(); return }
        if (!isProcessing.compareAndSet(false, true)) { image.close(); return }

        val startTime = System.currentTimeMillis()

        val imgW = image.width
        val imgH = image.height

        val grayMat = opticalFlowProcessor.extractGray(image)

        val bitmapBuffer = image.toBitmap()

        val matrix = Matrix().apply {
            postRotate(image.imageInfo.rotationDegrees.toFloat())
        }

        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )
        val detections = detector.detect(rotatedBitmap)

        val flowResult = opticalFlowProcessor.process(grayMat, detections, imgW, imgH)
        grayMat.release()

        val inferenceTime = System.currentTimeMillis() - startTime

        // ---- Confirmation logic ----
        val primaryDetection = detections.firstOrNull()
        if (primaryDetection != null) confirmedFrames++ else confirmedFrames = 0
        val confirmedDetection =
            if (confirmedFrames >= REQUIRED_CONFIRM_FRAMES) primaryDetection else null

        // ---- Audio update ----
        if (startTime - lastAudioUpdateTime > 150) {
            if (confirmedDetection != null) {
                if (!audioStarted) { audioEngine.nativeStart(); audioStarted = true }

                val cx = (confirmedDetection.x1 + confirmedDetection.x2) / 2f
                val rawAzimuth = (cx * 2f - 1f) * 60f
                smoothedAzimuth = 0.8f * smoothedAzimuth + 0.2f * rawAzimuth

                val width = confirmedDetection.x2 - confirmedDetection.x1
                val distance = max(0.5f, 2.0f - width)
                audioEngine.nativeSetSpatialParams(smoothedAzimuth, distance)
            } else {
                if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }
            }
            lastAudioUpdateTime = startTime
        }

        // ---- UI update ----
        runOnUiThread {
            overlayView.updateDetections(detections)
            overlayView.updateFlow(flowResult.points, imgW, imgH)

            if (System.currentTimeMillis() - lastUIUpdate > 200) {
                lastUIUpdate = System.currentTimeMillis()
                debugOverlay.text = if (confirmedDetection != null) {
                    val objFlow = flowResult.objectFlowMagnitudes[0] ?: 0f
                    getString(
                        R.string.debug_detection,
                        inferenceTime,
                        confirmedDetection.cnf,
                        smoothedAzimuth,
                        confirmedDetection.x2 - confirmedDetection.x1,
                        confirmedDetection.y2 - confirmedDetection.y1
                    ) + "\nflow=%.1fpx spd=%.2fm/s".format(objFlow, imuProcessor.speed)
                } else {
                    getString(R.string.debug_no_detection, inferenceTime) +
                            "\nspd=%.2fm/s".format(imuProcessor.speed)
                }
            }
        }

        image.close()
        isProcessing.set(false)
    }

    private fun startVideoTest(path: String) {
        prepareVideoTest()
        videoTestJob = cameraExecutor.submit { runVideoTest(path, null) }
    }

    private fun startVideoTestFromUri(uri: android.net.Uri) {
        prepareVideoTest()
        videoTestJob = cameraExecutor.submit { runVideoTest(null, uri) }
    }

    private fun prepareVideoTest() {
        // Stop camera if running
        if (systemRunning) stopSystem()

        videoRunning = true
        uploadButton.setText(R.string.stop)
        opticalFlowProcessor.release()
        confirmedFrames = 0
        previewView.visibility = View.INVISIBLE // no camera preview
        videoFrameView.visibility = View.VISIBLE
        debugOverlay.text = "Loading video..."
    }

    private fun stopVideoTest() {
        videoRunning = false
        videoTestJob?.cancel(true)
        videoTestJob = null
        uploadButton.setText(R.string.upload)
        if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }
        opticalFlowProcessor.release()
        confirmedFrames = 0
        runOnUiThread {
            videoFrameView.visibility = View.GONE
            overlayView.updateDetections(emptyList())
            overlayView.updateFlow(emptyList(), 1, 1)
            debugOverlay.setText(R.string.waiting)
        }
    }

    private fun runVideoTest(path: String?, uri: android.net.Uri?) {
        val retriever = MediaMetadataRetriever()
        if (path != null) retriever.setDataSource(path)
        else              retriever.setDataSource(this, uri)

        val durationMs = retriever
            .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLong() ?: run { retriever.release(); return }

        val frameIntervalMs = 1000L / 30L  // 30fps
        var timeMs = 0L
        var lastFrameRealTime = System.currentTimeMillis()

        while (videoRunning && timeMs < durationMs) {
            val frame = retriever.getFrameAtTime(
                timeMs * 1000L,
                MediaMetadataRetriever.OPTION_CLOSEST
            )
            if (frame == null) {
                timeMs += frameIntervalMs
                continue
            }

            runOnUiThread { videoFrameView.setImageBitmap(frame) }

            val imgW = frame.width
            val imgH = frame.height

            val grayMat = bitmapToGrayMat(frame)
            val detections = detector.detect(frame)
            val flowResult = opticalFlowProcessor.process(grayMat, detections, imgW, imgH)
            grayMat.release()

            if (detections.firstOrNull() != null) confirmedFrames++ else confirmedFrames = 0
            val confirmedDetection =
                if (confirmedFrames >= REQUIRED_CONFIRM_FRAMES) detections.firstOrNull() else null

            if (confirmedDetection != null) {
                if (!audioStarted) { audioEngine.nativeStart(); audioStarted = true }
                val cx = (confirmedDetection.x1 + confirmedDetection.x2) / 2f
                val rawAzimuth = (cx * 2f - 1f) * 60f
                smoothedAzimuth = 0.8f * smoothedAzimuth + 0.2f * rawAzimuth
                val width = confirmedDetection.x2 - confirmedDetection.x1
                audioEngine.nativeSetSpatialParams(smoothedAzimuth, maxOf(0.5f, 2.0f - width))
            } else {
                if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }
            }

            runOnUiThread {
                overlayView.updateDetections(detections)
                overlayView.updateFlow(flowResult.points, imgW, imgH)
                debugOverlay.text = if (confirmedDetection != null)
                    "${confirmedDetection.clsName} az=%.1f°".format(smoothedAzimuth)
                else
                    "no detection · frame ${flowResult.frameCount}"
            }

            val now = System.currentTimeMillis()
            val elapsed = now - lastFrameRealTime
            val framesToSkip = (elapsed / frameIntervalMs).coerceAtLeast(1)
            timeMs += frameIntervalMs * framesToSkip
            lastFrameRealTime = now
        }

        retriever.release()
        runOnUiThread { if (videoRunning) stopVideoTest() }
    }

    private fun bitmapToGrayMat(bitmap: android.graphics.Bitmap): org.opencv.core.Mat {
        val rgba = org.opencv.core.Mat()
        org.opencv.android.Utils.bitmapToMat(bitmap, rgba)
        val gray = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(rgba, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)
        rgba.release()
        return gray
    }

    // Resolves content:// Uri to a real file path where possible
    private fun getRealPathFromUri(uri: android.net.Uri): String? {
        var path: String? = null
        val projection = arrayOf(android.provider.MediaStore.Video.Media.DATA)
        contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val col = cursor.getColumnIndexOrThrow(android.provider.MediaStore.Video.Media.DATA)
                path = cursor.getString(col)
            }
        }
        return path
    }

    override fun onDestroy() {
        super.onDestroy()
        if (audioStarted) audioEngine.nativeStop()
        videoTestJob?.cancel(true)
        imuProcessor.stop()
        opticalFlowProcessor.release()
        cameraExecutor.shutdown()
    }

}