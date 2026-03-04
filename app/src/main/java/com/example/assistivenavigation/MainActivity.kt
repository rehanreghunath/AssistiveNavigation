package com.example.assistivenavigation

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
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
    private var wasDetecting = false
    private var lastAudioUpdateTime = 0L

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val isProcessing = AtomicBoolean(false)

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

        startButton.setOnClickListener {
            checkPermissionAndStart()
            audioEngine.nativeStart()
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

            val cameraProvider = cameraProviderFuture.get()

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
                    .build()

            val imageAnalysis =
                ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .build()

            imageAnalysis.setAnalyzer(
                cameraExecutor
            ) { image -> analyzeFrame(image) }

            val cameraSelector =
                CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()

            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {

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

        if (startTime - lastAudioUpdateTime > 150) {

            if (primaryDetection != null) {

                val cx =
                    (primaryDetection.x1 +
                            primaryDetection.x2) / 2f

                val normalized =
                    cx * 2f - 1f

                val rawAzimuth =
                    normalized * 60f

                smoothedAzimuth =
                    0.8f * smoothedAzimuth +
                            0.2f * rawAzimuth

                val width =
                    primaryDetection.x2 -
                            primaryDetection.x1

                val distance =
                    max(0.5f, 2.0f - width)

                audioEngine.nativeSetSpatialParams(
                    smoothedAzimuth,
                    distance
                )
            }

            lastAudioUpdateTime = startTime
        }

        runOnUiThread {

            overlayView.updateDetections(detections)

            if (System.currentTimeMillis() - lastUIUpdate > 200) {

                lastUIUpdate = System.currentTimeMillis()

                if (primaryDetection != null) {

                    val width =
                        primaryDetection.x2 -
                                primaryDetection.x1

                    val height =
                        primaryDetection.y2 -
                                primaryDetection.y1

                    debugOverlay.text =
                        getString(
                            R.string.debug_detection,
                            inferenceTime,
                            primaryDetection.score,
                            smoothedAzimuth,
                            width,
                            height
                        )

                    if (!wasDetecting) {
                        Log.d(
                            "YOLO_RESULT",
                            "Detection started"
                        )
                    }

                    wasDetecting = true

                } else {

                    debugOverlay.text =
                        getString(
                            R.string.debug_no_detection,
                            inferenceTime
                        )

                    if (wasDetecting) {
                        Log.d(
                            "YOLO_RESULT",
                            "Detection lost"
                        )
                    }

                    wasDetecting = false
                }
            }
        }

        image.close()
        isProcessing.set(false)
    }

    override fun onDestroy() {

        super.onDestroy()

        audioEngine.nativeStop()

        cameraExecutor.shutdown()
    }
}