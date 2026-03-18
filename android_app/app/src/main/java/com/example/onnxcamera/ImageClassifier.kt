package com.example.onnxcamera

import android.content.Context
import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

class ImageClassifier(private val context: Context) {
    private var env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val sessionLock = Any()
    private val labels = mutableListOf<String>()
    
    companion object {
        private const val ONNX_MODEL = "efficientnet_v2_s_int8.onnx"
        private const val QNN_CONTEXT_BINARY = "efficientnet_v2_s.bin"
        private const val LABEL_FILE = "labels.txt"
        private const val INPUT_SIZE = 224
        
        // ImageNet Normalization values
        private val IMAGE_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGE_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    data class Prediction(
        val index: Int,
        val label: String,
        val confidence: Float
    )

    data class ClassificationResult(
        val predictions: List<Prediction>,
        val latencyMs: Long,
        val preProcessMs: Long = 0,
        val postProcessMs: Long = 0
    ) {
        val totalMs: Long get() = latencyMs + preProcessMs + postProcessMs
        val fps: Double get() = if (totalMs > 0) 1000.0 / totalMs else 0.0
    }

    enum class Delegate {
        CPU, GPU, NNAPI, QNN
    }

    private var currentDelegate = Delegate.CPU
    private var isQnnCompiledMode = false

    init {
        // Load labels
        context.assets.open(LABEL_FILE).bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val parts = line.split(": ", limit = 2)
                if (parts.size == 2) {
                    labels.add(parts[1])
                } else {
                    labels.add(line)
                }
            }
        }
        
        prepareContextBinary()
        setDelegate(Delegate.CPU)
    }

    fun warmUp(iterations: Int = 5) {
        val dummyBitmap = createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        repeat(iterations) {
            classify(dummyBitmap)
        }
    }

    private fun prepareContextBinary() {
        val file = java.io.File(context.filesDir, QNN_CONTEXT_BINARY)
        if (!file.exists()) {
            context.assets.open(QNN_CONTEXT_BINARY).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
    }

    fun setQnnCompiledMode(enable: Boolean) {
        if (isQnnCompiledMode == enable) return
        isQnnCompiledMode = enable
        val tempDelegate = currentDelegate
        currentDelegate = Delegate.CPU 
        setDelegate(tempDelegate)
    }

    fun setDelegate(delegate: Delegate) {
        Log.i("AI_CLASSIFIER", "Transitioning to delegate: $delegate | Compiled Mode: $isQnnCompiledMode")
        try {
            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(4)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

            val newSession = if (isQnnCompiledMode && delegate == Delegate.QNN) {
                Log.i("AI_CLASSIFIER", "Attempting to load QNN HTP Context binary...")
                val qnnOptions = mutableMapOf<String, String>()
                qnnOptions["backend_path"] = "libQnnHtp.so"
                qnnOptions["library_path"] = context.applicationInfo.nativeLibraryDir
                qnnOptions["ep.context_file_path"] = java.io.File(context.filesDir, QNN_CONTEXT_BINARY).absolutePath
                
                try {
                    val addQnnMethod = options.javaClass.getMethod("addQnn", Map::class.java)
                    addQnnMethod.invoke(options, qnnOptions)
                } catch (e: Exception) {
                    Log.e("AI_CLASSIFIER", "QNN Reflection Binding Error: ${e.message}")
                }
                env.createSession(context.assets.open(ONNX_MODEL).readBytes(), options)
            } else {
                when (delegate) {
                    Delegate.CPU -> options.addConfigEntry("session.use_xnnpack", "1")
                    Delegate.GPU, Delegate.NNAPI -> options.addNnapi()
                    Delegate.QNN -> {
                        val qnnOptions = mutableMapOf<String, String>()
                        qnnOptions["backend_path"] = "libQnnHtp.so"
                        qnnOptions["library_path"] = context.applicationInfo.nativeLibraryDir
                        qnnOptions["htp_arch"] = "v75"
                        qnnOptions["htp_performance_mode"] = "high_performance"
                        try {
                            val addQnnMethod = options.javaClass.getMethod("addQnn", Map::class.java)
                            addQnnMethod.invoke(options, qnnOptions)
                        } catch (e: Exception) {
                           Log.e("AI_CLASSIFIER", "QNN Live EP Init Error: ${e.message}")
                        }
                    }
                }
                env.createSession(context.assets.open(ONNX_MODEL).readBytes(), options)
            }

            synchronized(sessionLock) {
                session?.close()
                session = newSession
                currentDelegate = delegate
            }
        } catch (e: Exception) {
            Log.e("AI_CLASSIFIER", "Hardware failure ($delegate): ${e.message}")
            if (delegate != Delegate.CPU) {
                 Log.w("AI_CLASSIFIER", "Falling back to CPU...")
                 setDelegate(Delegate.CPU)
            }
        }
    }

    fun getCurrentDelegate(): Delegate = currentDelegate
    
    fun classify(bitmap: Bitmap): ClassificationResult {
        val startTime = System.currentTimeMillis()
        
        // Null check for session
        val currentSession: OrtSession
        synchronized(sessionLock) {
            currentSession = session ?: return ClassificationResult(emptyList(), 0)
        }
        
        val scaledBitmap = if (bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE) 
            bitmap else bitmap.scale(INPUT_SIZE, INPUT_SIZE, true)
        
        val preTime = System.currentTimeMillis()
        val floatBuffer = preProcessImage(scaledBitmap)
        val preProcessMs = System.currentTimeMillis() - preTime

        val infTime = System.currentTimeMillis()
        val result = try {
            val inputName = currentSession.inputNames?.iterator()?.next() ?: "input"
            val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)
            
            val res = synchronized(sessionLock) {
                currentSession.run(Collections.singletonMap(inputName, inputTensor))
            }
            inputTensor.close()
            res
        } catch (e: Exception) {
            Log.e("AI_CLASSIFIER", "Inference error: ${e.message}")
            return ClassificationResult(emptyList(), 0)
        }
        
        val latencyMs = System.currentTimeMillis() - infTime

        val postTime = System.currentTimeMillis()
        @Suppress("UNCHECKED_CAST")
        val resultValue = result?.get(0)?.value
        val rawOutput = if (resultValue is Array<*> && resultValue[0] is FloatArray) {
            (resultValue as Array<FloatArray>)[0]
        } else {
            FloatArray(1000)
        }
        
        // Apply Softmax to convert logits to probabilities (%)
        val output = softmax(rawOutput)
        
        // Post-process to get Top-5
        val predictions = output.indices
            .map { i -> Prediction(i, if (i < labels.size) labels[i] else "Unknown ($i)", output[i]) }
            .sortedByDescending { it.confidence }
            .take(5)
        
        result?.close()
        val postProcessMs = System.currentTimeMillis() - postTime
        
        return ClassificationResult(predictions, latencyMs, preProcessMs, postProcessMs)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expSum = logits.fold(0f) { sum, logit -> sum + exp(logit - maxLogit) }
        return FloatArray(logits.size) { i -> exp(logits[i] - maxLogit) / expSum }
    }

    private fun preProcessImage(bitmap: Bitmap): FloatBuffer {
        val floatBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val pixelValue = pixels[i * INPUT_SIZE + j]

                val r = (pixelValue shr 16 and 0xFF) / 255.0f
                val g = (pixelValue shr 8 and 0xFF) / 255.0f
                val b = (pixelValue and 0xFF) / 255.0f

                val normR = (r - IMAGE_MEAN[0]) / IMAGE_STD[0]
                val normG = (g - IMAGE_MEAN[1]) / IMAGE_STD[1]
                val normB = (b - IMAGE_MEAN[2]) / IMAGE_STD[2]

                floatBuffer.put(i * INPUT_SIZE + j, normR)
                floatBuffer.put(INPUT_SIZE * INPUT_SIZE + i * INPUT_SIZE + j, normG)
                floatBuffer.put(2 * INPUT_SIZE * INPUT_SIZE + i * INPUT_SIZE + j, normB)
            }
        }
        floatBuffer.rewind()
        return floatBuffer
    }

    fun close() {
        session?.close()
        env.close()
    }
}
