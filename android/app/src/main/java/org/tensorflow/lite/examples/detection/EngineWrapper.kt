package org.tensorflow.lite.examples.detection

import android.content.res.AssetManager
import org.tensorflow.lite.examples.engine.FaceBox
import org.tensorflow.lite.examples.engine.FaceDetector
import org.tensorflow.lite.examples.engine.Live


class EngineWrapper(private var assetManager: AssetManager) {
    private var live: Live = Live()

    fun init(): Boolean {
        return live.loadModel(assetManager) == 0
    }

    fun destroy() {
        live.destroy()
    }

    fun detect(yuv: ByteArray, width: Int, height: Int, orientation: Int, box: FaceBox): DetectionResult {
        // 去掉了人脸检测，只进行活体检测
        if (box != null) {
            val begin = System.currentTimeMillis()
            box.confidence = detectLive(yuv, width, height, orientation, box)
            val end = System.currentTimeMillis()
            return DetectionResult(box, end - begin, true)
        }

        return DetectionResult()
    }

    private fun detectLive(
        yuv: ByteArray,
        width: Int,
        height: Int,
        orientation: Int,
        faceBox: FaceBox
    ): Float = live.detect(yuv, width, height, orientation, faceBox)

}