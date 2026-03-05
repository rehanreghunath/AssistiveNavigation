package com.example.assistivenavigation

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var startButton: Button
    private lateinit var overlayView: OverlayView
    private lateinit var debugOverlay: TextView

    private lateinit var detector: Detector
    private lateinit var audioEngine: AudioEngine

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
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) startCamera()
            else finish()
        }

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        debugOverlay = findViewById(R.id.debugOverlay)
        startButton = findViewById(R.id.startButton)

        detector = Detector(this)
        audioEngine = AudioEngine()

        previewView.visibility = View.INVISIBLE
        debugOverlay.setText(R.string.waiting)

        startButton.setOnClickListener {

            if (!systemRunning) {

                systemRunning = true
                startButton.setText(R.string.stop)

                previewView.visibility = View.VISIBLE
                debugOverlay.text = ""

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

        debugOverlay.setText(R.string.waiting)

        confirmedFrames = 0

        if (audioStarted) {
            audioEngine.nativeStop()
            audioStarted = false
        }
    }

    private fun checkPermissionAndStart() {

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) startCamera()
        else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(
                        previewView.surfaceProvider
                    )
                }

            val resolutionSelector =
                ResolutionSelector.Builder()
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

            val imageAnalysis =
                ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

            imageAnalysis.setAnalyzer(
                cameraExecutor
            ) { image -> analyzeFrame(image) }

            val cameraSelector =
                CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider?.unbindAll()

            cameraProvider?.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {

        if (!systemRunning) {
            image.close()
            return
        }

        if (!isProcessing.compareAndSet(false, true)) {
            image.close()
            return
        }

        val startTime = System.currentTimeMillis()

        val detections = detector.detect(image)

        val inferenceTime =
            System.currentTimeMillis() - startTime

        val primaryDetection =
            detections.firstOrNull()

        if (primaryDetection != null) {
            confirmedFrames++
        } else {
            confirmedFrames = 0
        }

        val confirmedDetection =
            if (confirmedFrames >= REQUIRED_CONFIRM_FRAMES)
                primaryDetection
            else null

        if (startTime - lastAudioUpdateTime > 150) {

            if (confirmedDetection != null) {

                if (!audioStarted) {
                    audioEngine.nativeStart()
                    audioStarted = true
                }

                val cx =
                    (confirmedDetection.x1 +
                            confirmedDetection.x2) / 2f

                val normalized =
                    cx * 2f - 1f

                val rawAzimuth =
                    normalized * 60f

                smoothedAzimuth =
                    0.8f * smoothedAzimuth +
                            0.2f * rawAzimuth

                val width =
                    confirmedDetection.x2 -
                            confirmedDetection.x1

                val distance =
                    max(0.5f, 2.0f - width)

                audioEngine.nativeSetSpatialParams(
                    smoothedAzimuth,
                    distance
                )

            } else {

                if (audioStarted) {
                    audioEngine.nativeStop()
                    audioStarted = false
                }
            }

            lastAudioUpdateTime = startTime
        }

        runOnUiThread {

            overlayView.updateDetections(detections)

            if (System.currentTimeMillis() - lastUIUpdate > 200) {

                lastUIUpdate = System.currentTimeMillis()

                if (confirmedDetection != null) {

                    val width =
                        confirmedDetection.x2 -
                                confirmedDetection.x1

                    val height =
                        confirmedDetection.y2 -
                                confirmedDetection.y1

                    debugOverlay.text =
                        getString(
                            R.string.debug_detection,
                            inferenceTime,
                            confirmedDetection.score,
                            smoothedAzimuth,
                            width,
                            height
                        )

                } else {

                    debugOverlay.text =
                        getString(
                            R.string.debug_no_detection,
                            inferenceTime
                        )
                }
            }
        }

        image.close()
        isProcessing.set(false)
    }

    override fun onDestroy() {

        super.onDestroy()

        if (audioStarted) {
            audioEngine.nativeStop()
        }

        cameraExecutor.shutdown()
    }
}