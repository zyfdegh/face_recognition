//
// Created by yuanhao on 20-6-9.
//

#ifndef LIVEBODYEXAMPLE_DEFINITION_H
#define LIVEBODYEXAMPLE_DEFINITION_H

#include <string>

#define JAVA_ARRAY_LIST_CLASSPATH           "java/util/ArrayList"
#define ANDROID_FACE_BOX_CLASSPATH          "com/mv/engine/FaceBox"

// 注意：如果要改 android 包名，要改下面两行，否则会出现
// Cannot resolve corresponding JNI function Java_<pkgname>_engine_Live_allocate
// 或者运行时报
// E/art: No implementation found for long <pkgname>.engine.Live.allocate()

#define FACE_DETECTOR_METHOD(METHOD_NAME) \
    Java_org_tensorflow_lite_examples_engine_FaceDetector_##METHOD_NAME

#define LIVE_METHOD(METHOD_NAME) \
    Java_org_tensorflow_lite_examples_engine_Live_##METHOD_NAME

struct FaceBox {
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};

struct ModelConfig {
    float scale;
    float shift_x;
    float shift_y;
    int height;
    int width;
    std::string name;
    bool org_resize;
};

#endif //LIVEBODYEXAMPLE_DEFINITION_H
