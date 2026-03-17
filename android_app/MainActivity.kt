package com.example.onnxcamera

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imageClassifier: ImageClassifier
    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the SNPE/NNAPI accelerated ONNX classifier
        imageClassifier = ImageClassifier(this)

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // CameraProvider
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Analysis
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(executor) { imageProxy ->
                        // 1. Convert ImageProxy to Bitmap (You need a standard util class for this YUV -> RGB conversion)
                        val bitmap = imageProxy.toBitmap()

                        // 2. Classify the bitmap using ONNX + DSP
                        val outputFloatArray = imageClassifier.classify(bitmap)

                        // 3. Find max index (Argmax) for EfficientNet prediction
                        var maxIndex = 0
                        var maxProb = 0.0f
                        for (i in outputFloatArray.indices) {
                            if (outputFloatArray[i] > maxProb) {
                                maxProb = outputFloatArray[i]
                                maxIndex = i
                            }
                        }

                        // Do something with output (Update UI, Toast, Logging, etc.)
                        Log.d("ONNX_NNAPI", "Predicted ImageNet Class Index: \$maxIndex with confidence: \$maxProb")

                        // Close frame to receive next
                        imageProxy.close()
                    }
                }

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageAnalyzer
                )

            } catch(exc: Exception) {
                Log.e("CAMERA_ERROR", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        imageClassifier.close()
        executor.shutdown()
    }
}
