package org.tensorflow.lite.examples.engine

import android.graphics.Rect
import android.graphics.RectF
import androidx.annotation.Keep

@Keep
data class FaceBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    var confidence: Float
)



fun NewFaceBoxFrom(rec: RectF): FaceBox {
    if (rec == null) {
        return FaceBox(0,0,0,0,0F);
    }
    return FaceBox(rec.left.toInt(), rec.top.toInt(), rec.right.toInt(), rec.bottom.toInt(), 0F);
}
