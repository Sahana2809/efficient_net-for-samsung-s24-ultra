package com.example.onnxcamera

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

class ImageClassifier(context: Context) {
    private var env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    
    // Model Constants
    private val MODEL_FILE = "efficientnet_v2_s_int8.onnx"
    private val INPUT_SIZE = 224
    private val NUM_CLASSES = 1000
    
    // ImageNet Normalization values
    private val IMAGE_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGE_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    // Delegate enum to select execution provider (CPU, GPU, NNAPI, QNN)
    enum class Delegate {
        CPU, GPU, NNAPI, QNN
    }

    private var currentDelegate = Delegate.CPU
    private val modelBytes by lazy { context.assets.open(MODEL_FILE).readBytes() }

    init {
        // Initialize with QNN for S24 Ultra / Snapdragon 8 Gen 3
        setDelegate(Delegate.QNN)
    }

    fun setDelegate(delegate: Delegate) {
        session?.close()
        val options = OrtSession.SessionOptions()

        when (delegate) {
            Delegate.CPU -> {
                // Default ONNX execution runs on CPU
            }
            Delegate.GPU -> {
                // NNAPI is often used for GPU acceleration on Android
                options.addNnapi() 
            }
            Delegate.NNAPI -> {
                // Standard NNAPI for broad compatibility
                options.addNnapi()
            }
            Delegate.QNN -> {
                // QNN provides direct access to the HTP on Snapdragon 8 Gen 3
                val qnnOptions = mutableMapOf<String, String>()
                qnnOptions["backend_path"] = "libQnnHtp.so"
                // For HTP v75 (S24 Ultra SoC), we explicitly set the backend
                options.addQnn(qnnOptions)
            }
        }

        session = env.createSession(modelBytes, options)
        currentDelegate = delegate
    }

    fun classify(bitmap: Bitmap): FloatArray {
        // 1. Resize the bitmap to match model input (224x224)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        // 2. Extract image data and normalize
        val floatBuffer = preProcessImage(scaledBitmap)

        // 3. Create input tensor [1, 3, 224, 224]
        val inputName = session?.inputNames?.iterator()?.next()
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)

        // 4. Run inference
        val result = session?.run(Collections.singletonMap(inputName, inputTensor))

        // 5. Parse output
        val output = (result?.get(0)?.value as Array<FloatArray>)[0]
        
        // Clean up resources for this run
        inputTensor.close()
        result?.close()
        
        return output
    }

    private fun preProcessImage(bitmap: Bitmap): FloatBuffer {
        // [1, 3, 224, 224] Shape requires flat float array
        val floatBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // ONNX models expect planar format NCHW (Channels, Height, Width)
        // rather than the standard Android NHWC (Height, Width, Channels)

        var pixelIndex = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val pixelValue = pixels[pixelIndex++]

                // Extract RGB
                val r = (pixelValue shr 16 and 0xFF) / 255.0f
                val g = (pixelValue shr 8 and 0xFF) / 255.0f
                val b = (pixelValue and 0xFF) / 255.0f

                // Normalize using ImageNet mean & std
                val normR = (r - IMAGE_MEAN[0]) / IMAGE_STD[0]
                val normG = (g - IMAGE_MEAN[1]) / IMAGE_STD[1]
                val normB = (b - IMAGE_MEAN[2]) / IMAGE_STD[2]

                // R channel (offset 0)
                floatBuffer.put(i * INPUT_SIZE + j, normR)
                // G channel (offset 224*224)
                floatBuffer.put(INPUT_SIZE * INPUT_SIZE + i * INPUT_SIZE + j, normG)
                // B channel (offset 2*224*224)
                floatBuffer.put(2 * INPUT_SIZE * INPUT_SIZE + i * INPUT_SIZE + j, normB)
            }
        }
        
        return floatBuffer
    }

    fun close() {
        session?.close()
        env.close()
    }
}
