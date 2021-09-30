package org.tensorflow.lite.examples.engine

import android.graphics.Rect
import androidx.annotation.Keep

@Keep
data class FaceBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    var confidence: Float
)

fun NewFaceBoxFrom(rec: Rect): FaceBox {
    if (rec == null) {
        return FaceBox(0,0,0,0,0F);
    }
    return FaceBox(rec.left, rec.top, rec.right, rec.bottom, 0F);
}