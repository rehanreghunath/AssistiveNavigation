package com.example.assistivenavigation

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.os.Bundle
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

class MainActivity : AppCompatActivity() {

    // ── Views ─────────────────────────────────────────────────────────────────
    private lateinit var previewView:    PreviewView
    private lateinit var startButton:    Button
    private lateinit var uploadButton:   Button
    private lateinit var overlayView:    OverlayView
    private lateinit var debugOverlay:   TextView
    private lateinit var videoFrameView: android.widget.ImageView

    // ── Pipeline components ───────────────────────────────────────────────────
    private lateinit var detector:            Detector
    private lateinit var boxTracker:          BoxTracker
    private lateinit var audioEngine:         AudioEngine
    private lateinit var distanceEstimator:   DistanceEstimator
    private lateinit var opticalFlowProcessor: OpticalFlowProcessor
    private lateinit var imuProcessor:        IMUProcessor

    // ── Session state ─────────────────────────────────────────────────────────
    private var systemRunning  = false
    private var audioStarted   = false
    private var videoRunning   = false

    // ── Audio / UI timing ─────────────────────────────────────────────────────
    // @Volatile so the camera thread and UI thread both see the latest value.
    // smoothedAzimuth is only ever written on the camera thread, but accessed
    // as a snapshot before each runOnUiThread call, so @Volatile is enough.
    @Volatile private var smoothedAzimuth    = 0f
    private var lastUIUpdate         = 0L
    private var lastAudioUpdateTime  = 0L

    // ── Concurrency ───────────────────────────────────────────────────────────
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val isProcessing   = AtomicBoolean(false)

    private var cameraProvider:  ProcessCameraProvider?          = null
    private var videoTestJob:    java.util.concurrent.Future<*>? = null

    // ── Permission / file launchers ───────────────────────────────────────────
    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else finish()
        }

    private val videoPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            uri ?: return@registerForActivityResult
            val path = getRealPathFromUri(uri)
            if (path != null) startVideoTest(path)
            else              startVideoTestFromUri(uri)
        }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        OpenCVLoader.initLocal()
        setContentView(R.layout.activity_main)

        // Bind all views before any logic touches them
        previewView    = findViewById(R.id.previewView)
        overlayView    = findViewById(R.id.overlayView)
        debugOverlay   = findViewById(R.id.debugOverlay)
        startButton    = findViewById(R.id.startButton)
        uploadButton   = findViewById(R.id.uploadButton)
        videoFrameView = findViewById(R.id.videoFrameView)

        // Initialise pipeline components
        detector             = Detector(baseContext, MODEL_PATH, LABELS_PATH)
        boxTracker           = BoxTracker()
        audioEngine          = AudioEngine()
        opticalFlowProcessor = OpticalFlowProcessor()
        imuProcessor         = IMUProcessor(this)
        distanceEstimator    = DistanceEstimator()

        previewView.visibility = View.INVISIBLE
        debugOverlay.setText(R.string.waiting)

        uploadButton.setOnClickListener {
            if (!videoRunning) videoPicker.launch("video/*")
            else               stopVideoTest()
        }

        startButton.setOnClickListener {
            if (!systemRunning) {
                systemRunning = true
                startButton.setText(R.string.stop)
                previewView.visibility = View.VISIBLE
                debugOverlay.text = ""
                imuProcessor.start()
                checkPermissionAndStart()
            } else {
                stopSystem()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (audioStarted) audioEngine.nativeStop()
        videoTestJob?.cancel(true)
        imuProcessor.stop()
        opticalFlowProcessor.release()
        distanceEstimator.reset()
        boxTracker.reset()
        cameraExecutor.shutdown()
    }

    // ── Camera session ────────────────────────────────────────────────────────

    private fun stopSystem() {
        systemRunning = false
        startButton.setText(R.string.start)

        cameraProvider?.unbindAll()
        previewView.visibility = View.INVISIBLE
        overlayView.updateDetections(emptyList())
        overlayView.updateFlow(emptyList(), 1, 1)
        debugOverlay.setText(R.string.waiting)

        if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }

        imuProcessor.stop()
        opticalFlowProcessor.release()
        distanceEstimator.reset()
        boxTracker.reset()

        // Reset smoothing state so next session starts clean
        smoothedAzimuth     = 0f
        lastAudioUpdateTime = 0L
        lastUIUpdate        = 0L
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
                .setAspectRatioStrategy(
                    AspectRatioStrategy(
                        AspectRatio.RATIO_16_9,
                        AspectRatioStrategy.FALLBACK_RULE_AUTO
                    )
                )
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(1280, 720),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
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

    // ── Frame analysis (camera thread) ────────────────────────────────────────

    private fun analyzeFrame(image: ImageProxy) {
        if (!systemRunning)                           { image.close(); return }
        if (!isProcessing.compareAndSet(false, true)) { image.close(); return }

        val startTime = System.currentTimeMillis()
        val imgW      = image.width
        val imgH      = image.height

        // ── 1. Extract grayscale for optical flow (reads Y plane directly,
        //       no full-colour conversion needed)
        val grayMat = opticalFlowProcessor.extractGray(image)

        // ── 2. Convert to bitmap and rotate to portrait.
        //       Both bitmaps are recycled immediately after use to avoid
        //       native memory pressure at 30fps.
        val rawBitmap = image.toBitmap()
        val matrix    = Matrix().apply {
            postRotate(image.imageInfo.rotationDegrees.toFloat())
        }
        val rotatedBitmap = Bitmap.createBitmap(
            rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true
        )
        rawBitmap.recycle()   // free source; rotatedBitmap is the one we need

        // ── 3. YOLO detection
        val yoloStartTime = System.currentTimeMillis()
        val detections    = detector.detect(rotatedBitmap)
        val inferenceMs   = System.currentTimeMillis() - yoloStartTime
        rotatedBitmap.recycle()   // done with bitmap

        // ── 4. Optical flow (uses grayMat from step 1)
        val flowResult = opticalFlowProcessor.process(grayMat, detections, imgW, imgH)
        grayMat.release()   // MUST release; OpenCV native memory is not GC-managed
        val totalPipelineMs = System.currentTimeMillis() - startTime

        // ── 5. Box tracker — produces a stable confirmed box or null
        val confirmedDetection = boxTracker.update(detections)

        // Snapshot tracker state on the camera thread before handing off to UI thread.
        // Reading boxTracker.isCoasting / boxTracker.state from the UI thread without
        // a snapshot would be a data race.
        val isCoasting   = boxTracker.isCoasting
        val trackerState = boxTracker.state.name

        // ── 6. Find the flow magnitude for the TRACKED object specifically.
        //       objectFlowMagnitudes is keyed by index in the raw detections list.
        //       When coasting, the predicted box is not in the list at all.
        val trackedIndex = if (confirmedDetection != null && !isCoasting)
            detections.indexOf(confirmedDetection)
        else -1

        val objFlow = if (trackedIndex >= 0)
            flowResult.objectFlowMagnitudes[trackedIndex] ?: 0f
        else
        // Coasting: use whatever flow was computed for the highest-priority
        // detection as a best-effort estimate
            flowResult.objectFlowMagnitudes[0] ?: 0f

        val bgFlow = flowResult.backgroundFlowMagnitude

        // ── 7. Audio update (throttled to every 150ms to avoid hammering Oboe)
        if (startTime - lastAudioUpdateTime > 150) {
            if (confirmedDetection != null) {
                if (!audioStarted) { audioEngine.nativeStart(); audioStarted = true }

                val rawAzimuth  = (confirmedDetection.cx * 2f - 1f) * 90f
                smoothedAzimuth = 0.85f * smoothedAzimuth + 0.15f * rawAzimuth

                val distance = distanceEstimator.estimate(
                    objectFlowMag     = objFlow,
                    backgroundFlowMag = bgFlow,
                    imuSpeed          = imuProcessor.speed,
                    bboxWidth         = confirmedDetection.w
                )
                audioEngine.nativeSetSpatialParams(smoothedAzimuth, distance)
            } else {
                if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }
                // Don't let the old smoothed distance bleed into the next detection
                distanceEstimator.reset()
            }
            lastAudioUpdateTime = startTime
        }

        // Snapshot after the audio block so the UI sees the post-smoothing value
        val azimuthSnapshot = smoothedAzimuth

        // ── 8. UI update (throttled to every 200ms to reduce main-thread work)
        runOnUiThread {
            overlayView.updateDetections(
                if (confirmedDetection != null) listOf(confirmedDetection) else emptyList(),
                isCoasting   // snapshot from camera thread — no race
            )
            overlayView.updateFlow(flowResult.points, imgW, imgH)

            if (System.currentTimeMillis() - lastUIUpdate > 200) {
                lastUIUpdate = System.currentTimeMillis()

                debugOverlay.text = if (confirmedDetection != null) {
                    getString(
                        R.string.debug_detection,
                        inferenceMs,
                        confirmedDetection.cnf,
                        azimuthSnapshot,
                        confirmedDetection.w,
                        confirmedDetection.h
                    ) + "\nbg=%.1fpx obj=%.1fpx spd=%.2fm/s [%s] pipe=%dms".format(
                        bgFlow,
                        objFlow,
                        imuProcessor.speed,
                        if (isCoasting) "COAST" else "TRACK",
                        totalPipelineMs
                    )
                } else {
                    getString(R.string.debug_no_detection, inferenceMs) +
                            "\nspd=%.2fm/s [%s] pipe=%dms".format(
                                imuProcessor.speed,
                                trackerState,   // shows IDLE or CONFIRMING
                                totalPipelineMs
                            )
                }
            }
        }

        image.close()
        isProcessing.set(false)
    }

    // ── Video test ────────────────────────────────────────────────────────────

    private fun startVideoTest(path: String) {
        prepareVideoTest()
        videoTestJob = cameraExecutor.submit { runVideoTest(path, null) }
    }

    private fun startVideoTestFromUri(uri: android.net.Uri) {
        prepareVideoTest()
        videoTestJob = cameraExecutor.submit { runVideoTest(null, uri) }
    }

    private fun prepareVideoTest() {
        if (systemRunning) stopSystem()   // stopSystem() handles its own resets

        // Always reset regardless of whether the camera was running, because the
        // user may tap UPLOAD after a previous video without going through stopSystem
        distanceEstimator.reset()
        boxTracker.reset()
        smoothedAzimuth     = 0f
        lastAudioUpdateTime = 0L
        lastUIUpdate        = 0L

        videoRunning = true
        uploadButton.setText(R.string.stop)
        opticalFlowProcessor.release()
        previewView.visibility     = View.INVISIBLE
        videoFrameView.visibility  = View.VISIBLE
        debugOverlay.text          = "Loading video..."
    }

    private fun stopVideoTest() {
        videoRunning = false
        videoTestJob?.cancel(true)
        videoTestJob = null
        uploadButton.setText(R.string.upload)

        if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }
        opticalFlowProcessor.release()
        distanceEstimator.reset()
        boxTracker.reset()
        smoothedAzimuth     = 0f
        lastAudioUpdateTime = 0L
        lastUIUpdate        = 0L

        runOnUiThread {
            videoFrameView.visibility = View.GONE
            overlayView.updateDetections(emptyList())
            overlayView.updateFlow(emptyList(), 1, 1)
            debugOverlay.setText(R.string.waiting)
        }
    }

    private fun runVideoTest(path: String?, uri: android.net.Uri?) {
        val retriever = MediaMetadataRetriever()
        try {
            if (path != null) retriever.setDataSource(path)
            else              retriever.setDataSource(this, uri)
        } catch (e: Exception) {
            runOnUiThread { debugOverlay.text = "Failed to open video" }
            retriever.release()
            return
        }

        val durationMs = retriever
            .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLong() ?: run { retriever.release(); return }

        val frameIntervalMs   = 1000L / 30L
        var timeMs            = 0L
        var lastFrameRealTime = System.currentTimeMillis()

        while (videoRunning && timeMs < durationMs) {

            val frame = retriever.getFrameAtTime(
                timeMs * 1000L,
                MediaMetadataRetriever.OPTION_CLOSEST
            )
            if (frame == null) { timeMs += frameIntervalMs; continue }

            runOnUiThread { videoFrameView.setImageBitmap(frame) }

            val imgW = frame.width
            val imgH = frame.height

            // ── Pipeline ────────────────────────────────────────────────────
            val yoloStart  = System.currentTimeMillis()
            val grayMat    = bitmapToGrayMat(frame)
            val detections = detector.detect(frame)
            val inferenceMs = System.currentTimeMillis() - yoloStart

            val flowResult = opticalFlowProcessor.process(grayMat, detections, imgW, imgH)
            grayMat.release()
            val totalMs = System.currentTimeMillis() - yoloStart

            // Note: video frames from MediaMetadataRetriever are managed by the
            // retriever; do NOT call frame.recycle() here.

            // ── Tracker ─────────────────────────────────────────────────────
            val confirmedDetection = boxTracker.update(detections)
            val isCoasting         = boxTracker.isCoasting
            val trackerState       = boxTracker.state.name

            // ── Flow for tracked object ──────────────────────────────────────
            val trackedIndex = if (confirmedDetection != null && !isCoasting)
                detections.indexOf(confirmedDetection) else -1
            val objFlow = if (trackedIndex >= 0)
                flowResult.objectFlowMagnitudes[trackedIndex] ?: 0f
            else
                flowResult.objectFlowMagnitudes[0] ?: 0f
            val bgFlow = flowResult.backgroundFlowMagnitude

            // ── Audio ────────────────────────────────────────────────────────
            if (confirmedDetection != null) {
                if (!audioStarted) { audioEngine.nativeStart(); audioStarted = true }

                val rawAzimuth  = (confirmedDetection.cx * 2f - 1f) * 90f
                smoothedAzimuth = 0.85f * smoothedAzimuth + 0.15f * rawAzimuth

                val distance = distanceEstimator.estimate(
                    objectFlowMag     = objFlow,
                    backgroundFlowMag = bgFlow,
                    imuSpeed          = imuProcessor.speed,
                    bboxWidth         = confirmedDetection.w
                )
                audioEngine.nativeSetSpatialParams(smoothedAzimuth, distance)
            } else {
                if (audioStarted) { audioEngine.nativeStop(); audioStarted = false }
                distanceEstimator.reset()
            }

            val azimuthSnapshot = smoothedAzimuth

            // ── UI ───────────────────────────────────────────────────────────
            runOnUiThread {
                overlayView.updateDetections(
                    if (confirmedDetection != null) listOf(confirmedDetection) else emptyList(),
                    isCoasting
                )
                overlayView.updateFlow(flowResult.points, imgW, imgH)

                debugOverlay.text = if (confirmedDetection != null)
                    "${confirmedDetection.clsName} az=%.1f° obj=%.1fpx bg=%.1fpx [%s] %dms/%dms"
                        .format(
                            azimuthSnapshot,
                            objFlow,
                            bgFlow,
                            if (isCoasting) "COAST" else "TRACK",
                            inferenceMs,
                            totalMs
                        )
                else
                    "no detection · $trackerState · frame ${flowResult.frameCount} · ${totalMs}ms"
            }

            // ── Frame pacing ─────────────────────────────────────────────────
            // Skip frames if processing took longer than one frame interval so
            // the video doesn't fall further and further behind real time.
            val now          = System.currentTimeMillis()
            val elapsed      = now - lastFrameRealTime
            val framesToSkip = (elapsed / frameIntervalMs).coerceAtLeast(1)
            timeMs          += frameIntervalMs * framesToSkip
            lastFrameRealTime = now
        }

        retriever.release()
        runOnUiThread { if (videoRunning) stopVideoTest() }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Converts a Bitmap to a single-channel grayscale OpenCV Mat.
     * Used only in video-test mode; the camera path uses extractGray() directly
     * from the YUV Y-plane, which is faster.sssssssss
     */
    private fun bitmapToGrayMat(bitmap: Bitmap): org.opencv.core.Mat {
        val rgba = org.opencv.core.Mat()
        org.opencv.android.Utils.bitmapToMat(bitmap, rgba)
        val gray = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(rgba, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)
        rgba.release()
        return gray
    }

    /**
     * Resolves a content:// Uri to a real filesystem path where the OS allows it.
     * Returns null if the Uri cannot be resolved (e.g. cloud-backed files), in
     * which case the caller falls back to setDataSource(Uri).
     */
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
}