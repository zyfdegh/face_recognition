
# Real time face recognition in Android 


[![example 1](http://img.youtube.com/vi/V1IflHa8AAY/0.jpg)](https://youtu.be/V1IflHa8AAY "demo 1")
[![example 2](http://img.youtube.com/vi/y-1lO3m-SRI/0.jpg)](https://youtu.be/y-1lO3m-SRI "demo 2")


本 fork 改动点
1. 支持一人多脸，以适应不同角度、不同穿戴的图片
2. 支持存储每张添加的图片到 SD 卡，并按标签存放到目录
3. 未检测到人脸时，不允许添加，添加 toast 提示
4. 修复添加人脸时，正面相机脸部溢出屏幕左侧引起的闪退（改为 toast 提示）
5. 修复添加人脸时，正面相机展示错误照片（水平翻转）
6. 修复安卓 4.4 之后，存储授权问题
7. 替换应用名与图标
8. 升级 gradle 到 7.x 版本