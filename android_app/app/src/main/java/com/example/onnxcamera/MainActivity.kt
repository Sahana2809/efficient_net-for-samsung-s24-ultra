package com.example.onnxcamera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import android.widget.ToggleButton
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.InputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imageClassifier: ImageClassifier
    private val executor = Executors.newSingleThreadExecutor()
    private var isUpdatingUI = false
    
    // UI Elements
    private lateinit var resultText: TextView
    private lateinit var metricsText: TextView
    private lateinit var comparisonText: TextView
    private lateinit var viewFinder: PreviewView
    private lateinit var modelSwitch: ToggleButton
    private lateinit var backendGroup: RadioGroup
    private lateinit var referenceImage: ImageView
    private lateinit var groundTruthText: TextView
    private lateinit var btnNext: Button
    private lateinit var btnBenchmark: Button
    private lateinit var top5Text: TextView

    // Dataset evaluation state
    private val datasetImages = mutableListOf<String>()
    private var currentImageIndex = 0
    private var currentGroundTruthLabel = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind UI
        resultText = findViewById(R.id.resultText)
        metricsText = findViewById(R.id.metricsText)
        comparisonText = findViewById(R.id.comparisonText)
        viewFinder = findViewById(R.id.viewFinder)
        modelSwitch = findViewById(R.id.modelSwitch)
        backendGroup = findViewById(R.id.backendGroup)
        referenceImage = findViewById(R.id.referenceImage)
        groundTruthText = findViewById(R.id.groundTruthText)
        btnNext = findViewById(R.id.btnNext)
        btnBenchmark = findViewById(R.id.btnBenchmark)
        top5Text = findViewById(R.id.top5PredictionsText)

        imageClassifier = ImageClassifier(this)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        
        loadDatasetList()
        setupListeners()
        showNextImage()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private fun loadDatasetList() {
        try {
            val files = assets.list("images") ?: emptyArray()
            datasetImages.addAll(files.filter { it.endsWith(".jpg") || it.endsWith(".png") })
        } catch (e: Exception) {
            Log.e("DATASET_ERROR", "Failed to load assets/images", e)
        }
    }

    private fun showNextImage() {
        if (datasetImages.isEmpty()) return
        
        val fileName = datasetImages[currentImageIndex]
        // Filename format: targetIndex_labelName.jpg
        val parts = fileName.split("_")
        if (parts.isNotEmpty()) {
            val targetLabelIndex = parts[0].toIntOrNull() ?: -1
            // We'll use the filename for ground truth display
            currentGroundTruthLabel = fileName.substringAfter("_").substringBefore(".")
            groundTruthText.text = "GT: $currentGroundTruthLabel"
            
            val inputStream: InputStream = assets.open("images/$fileName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            referenceImage.setImageBitmap(bitmap)
        }
        
        currentImageIndex = (currentImageIndex + 1) % datasetImages.size
    }

    private fun setupListeners() {
        backendGroup.setOnCheckedChangeListener { _, checkedId ->
            if (isUpdatingUI) return@setOnCheckedChangeListener
            
            val delegate = when (checkedId) {
                R.id.btnCpu -> ImageClassifier.Delegate.CPU
                R.id.btnGpu -> ImageClassifier.Delegate.NNAPI
                R.id.btnNpu -> ImageClassifier.Delegate.QNN
                else -> ImageClassifier.Delegate.CPU
            }
            
            executor.execute {
                try {
                    imageClassifier.setDelegate(delegate)
                    
                    val actualDelegate = imageClassifier.getCurrentDelegate()
                    runOnUiThread {
                        if (actualDelegate != delegate) {
                            isUpdatingUI = true
                            backendGroup.check(R.id.btnCpu)
                            isUpdatingUI = false
                            Toast.makeText(this, "${delegate.name} fallback to CPU.", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    Log.e("MAIN", "Error setting delegate: ${e.message}")
                    runOnUiThread {
                        isUpdatingUI = true
                        backendGroup.check(R.id.btnCpu)
                        isUpdatingUI = false
                        Toast.makeText(this, "Backend error.", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        modelSwitch.setOnCheckedChangeListener { _, isChecked ->
            executor.execute {
                imageClassifier.setQnnCompiledMode(isChecked)
                val current = imageClassifier.getCurrentDelegate()
                if (current == ImageClassifier.Delegate.QNN) {
                    imageClassifier.setDelegate(ImageClassifier.Delegate.QNN)
                }
            }
        }

        btnNext.setOnClickListener {
            showNextImage()
            comparisonText.visibility = View.GONE
        }

        btnBenchmark.setOnClickListener {
            runDatasetBenchmark()
        }
    }

    private fun runDatasetBenchmark() {
        executor.execute {
            // Warm-up to ensure stable performance
            runOnUiThread { resultText.text = "Warming up..." }
            imageClassifier.warmUp(10)

            var top1Correct = 0
            var top5Correct = 0
            var totalLatency = 0L
            val total = datasetImages.size
            if (total == 0) return@execute
            
            runOnUiThread {
                resultText.text = "Benchmarking $total images..."
                comparisonText.visibility = View.VISIBLE
            }

            var maxLatency = 0L
            var minLatency = Long.MAX_VALUE
            
            datasetImages.forEach { fileName ->
                // Expected format: index_label.jpg
                val parts = fileName.split("_")
                val expectedIndex = parts[0].toIntOrNull() ?: -1
                
                val inputStream: InputStream = assets.open("images/$fileName")
                val bitmap = BitmapFactory.decodeStream(inputStream)
                
                val result = imageClassifier.classify(bitmap)
                totalLatency += result.latencyMs
                
                if (result.latencyMs > maxLatency) maxLatency = result.latencyMs
                if (result.latencyMs < minLatency) minLatency = result.latencyMs

                // Top-1 Accuracy
                if (result.predictions.isNotEmpty() && result.predictions[0].index == expectedIndex) {
                    top1Correct++
                }

                // Top-5 Accuracy
                if (result.predictions.any { it.index == expectedIndex }) {
                    top5Correct++
                }
            }

            val top1Acc = (top1Correct.toFloat() / total) * 100
            val top5Acc = (top5Correct.toFloat() / total) * 100
            val avgLatency = totalLatency / total

            val gflops = 8.8
            val avgTops = if (avgLatency > 0) (gflops / avgLatency) else 0.0
            val avgFps = if (totalLatency > 0) (1000.0 * total / totalLatency) else 0.0

            runOnUiThread {
                resultText.text = "Report: ${imageClassifier.getCurrentDelegate().name}"
                metricsText.text = "Avg:${avgLatency}ms | Max:${maxLatency} | Min:${if (minLatency == Long.MAX_VALUE) 0 else minLatency}\n" +
                        "Avg TOPS: ${String.format("%.2f", avgTops)} | Avg FPS: ${String.format("%.1f", avgFps)}"
                comparisonText.text = "Acc: Top-1: ${String.format("%.1f", top1Acc)}% | Top-5: ${String.format("%.1f", top5Acc)}%"
                comparisonText.setTextColor(0xFF03DAC5.toInt())
                comparisonText.visibility = View.VISIBLE
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = androidx.camera.core.Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(executor) { imageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        val result = imageClassifier.classify(bitmap)

                        runOnUiThread {
                            if (result.predictions.isNotEmpty()) {
                                val topResult = result.predictions[0]
                                resultText.text = "${topResult.label}\n${String.format("%.2f", topResult.confidence * 100)}%"
                                
                                // Show Top-5 list
                                val top5Str = result.predictions.joinToString("\n") { 
                                    "${it.label}: ${String.format("%.1f", it.confidence * 100)}%" 
                                }
                                top5Text.text = top5Str
                                
                                val activeBackend = imageClassifier.getCurrentDelegate().name
                                val gflops = 8.8 
                                val tops = if (result.latencyMs > 0) (gflops / result.latencyMs) else 0.0
                                
                                metricsText.text = "Pre:${result.preProcessMs} | Inf:${result.latencyMs} | Post:${result.postProcessMs}ms\n" +
                                        "FPS: ${String.format("%.1f", result.fps)} | $activeBackend | ${String.format("%.2f", tops)} TOPS"
                                
                                // Real-time comparison with reference image (Top-1)
                                val expectedLabel = currentGroundTruthLabel.lowercase()
                                val predictedLabel = topResult.label.lowercase()
                                if (predictedLabel.contains(expectedLabel) || expectedLabel.contains(predictedLabel)) {
                                    comparisonText.text = "MATCH ✅"
                                    comparisonText.setTextColor(0xFF00FF00.toInt())
                                } else {
                                    comparisonText.text = "MISMATCH ❌ (Expected: $currentGroundTruthLabel)"
                                    comparisonText.setTextColor(0xFFFF0000.toInt())
                                }
                                comparisonText.visibility = View.VISIBLE
                            }
                        }

                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
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
