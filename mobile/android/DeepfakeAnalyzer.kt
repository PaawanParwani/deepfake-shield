// DeepfakeAnalyzer.kt
// Native Android CameraX ImageAnalysis analyzer using ONNX Runtime Mobile with NNAPI acceleration.

package com.antigravity.deepfake.detector

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ai.onnxruntime.*
import java.nio.FloatBuffer

class DeepfakeAnalyzer(
    context: Context,
    private val onResult: (Boolean, Float, Long) -> Unit
) : ImageAnalysis.Analyzer {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    init {
        val modelBytes = context.assets.open("deepfake_detector_sota.onnx").readBytes()
        val sessionOptions = OrtSession.SessionOptions().apply {
            // Enable hardware acceleration (NPU / GPU via Android NNAPI)
            addNnapi()
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        ortSession = ortEnv.createSession(modelBytes, sessionOptions)
    }

    override fun analyze(imageProxy: ImageProxy) {
        val startTime = System.currentTimeMillis()
        val bitmap = imageProxy.toBitmap() // 1280x720 frame
        
        // Scale to 224x224
        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        
        // Allocate FloatBuffer for NCHW (1, 3, 224, 224)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        val pixels = IntArray(224 * 224)
        scaled.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        
        // Populate R, G, B channels with ImageNet normalization
        for (c in 0..2) {
            for (i in pixels.indices) {
                val p = pixels[i]
                val colorVal = when (c) {
                    0 -> ((p shr 16) and 0xFF) / 255.0f
                    1 -> ((p shr 8) and 0xFF) / 255.0f
                    else -> (p and 0xFF) / 255.0f
                }
                floatBuffer.put((colorVal - mean[c]) / std[c])
            }
        }
        floatBuffer.rewind()

        // Inference
        val inputName = ortSession.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1, 3, 224, 224))
        
        ortSession.run(mapOf(inputName to tensor)).use { results ->
            @Suppress("UNCHECKED_CAST")
            val outputArray = results[0].value as Array<FloatArray>
            val logits = outputArray[0]
            
            // Softmax
            val exp0 = Math.exp(logits[0].toDouble())
            val exp1 = Math.exp(logits[1].toDouble())
            val pFake = (exp1 / (exp0 + exp1)).toFloat()
            
            val isFake = pFake >= 0.65f
            val latency = System.currentTimeMillis() - startTime
            
            onResult(isFake, pFake, latency)
        }

        imageProxy.close()
    }
}
