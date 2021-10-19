/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.gson.Gson;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.tensorflow.lite.examples.engine.FaceBox;

import kotlin.experimental.ExperimentalTypeInference;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();


  // FaceNet
//  private static final int TF_OD_API_INPUT_SIZE = 160;
//  private static final boolean TF_OD_API_IS_QUANTIZED = false;
//  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
//  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";

  // MobileFaceNet
  private static final int TF_OD_API_INPUT_SIZE = 112;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;

  // image setting
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final Size LIVE_INPUT_SIZE = new Size(640, 480);
  //private static final int CROP_SIZE = 320;
  //private static final Size CROP_SIZE = new Size(320, 320);

  // bluetooth
  private static boolean btConnecting = false;
  private static boolean btWriteError = false;
  private static final String BT_CMD_ON = "A00101A2";
  private static final String BT_CMD_OFF = "A00100A1";
  public static final int MESSAGE_READ = 0;
  public static final int MESSAGE_WRITE = 1;
  public static final int MESSAGE_TOAST = 2;
  private BluetoothAdapter btAdapter = null;
  private BluetoothSocket btSocket = null;
  private static final String BT_MAC_ADDR_HARDCODE = "D8:76:E8:EC:B6:01";
  private static final String BT_UUID_HARDCODE = "00001101-0000-1000-8000-00805F9B34FB";
  private BluetoothSocket btSock = null;

  // 用来防止同一个人脸，不停被识别，不停开门的情况（只开一次）
  private static String lastRecognizedLabel = "";

  // 存储人脸图
  private static final boolean SAVE_FACE_WHEN_ADD = true;
  private static final String FACE_EMBEDDING_SUFFIX = ".embeddings";

  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private static final String TAG = "DetectorActivity";

  // 人脸识别
  private SimilarityClassifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap liveInputImg = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;
  private boolean canAddFace = false;
  private boolean faceDetected = false;
  //private boolean adding = false;

  private long imageCounter = 0;

  private Matrix frameToCropTransform;
  private Matrix frameToLiveTransform;
  private Matrix cropToFrameTransform;
  //private Matrix cropToPortraitTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // 人脸检测
  private FaceDetector faceDetector;

  // here the preview image is drawn in portrait way
  private Bitmap portraitBmp = null;
  // here the face is cropped and drawn
  private Bitmap faceBmp = null;

  private FloatingActionButton fabAdd;
  private FloatingActionButton fabSetting;
  private ListView lvSettings;

//  private HashMap<String, Classifier.Recognition> knownFaces = new HashMap<>();

  // 活体检测
  private boolean liveFaceEnginePrepared = false;
  private EngineWrapper liveFaceEngine;

  // FIXME 使用 Preference 存储
  private static String settingPassword = "";
  private static Float settingFacerecThreshold = 0.8F;
  private static Float settingLivebodyThreshold = 0.915F;

  @Override
  public synchronized void onResume() {
    liveFaceEngine = new EngineWrapper(getAssets());
    liveFaceEnginePrepared = liveFaceEngine.init();
    if (!liveFaceEnginePrepared) {
      Toast.makeText(this, "Engine init failed.", Toast.LENGTH_LONG).show();
    }
    super.onResume();
  }

  @Override
  public synchronized void onDestroy() {
    liveFaceEngine.destroy();
    if (btSock != null) {
      try {
          btSock.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    super.onDestroy();
  }


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    fabAdd = findViewById(R.id.fab_add);
    fabAdd.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        onAddClick();
      }
    });

    fabSetting = findViewById(R.id.fab_setting);
    fabSetting.setOnClickListener(new View.OnClickListener(){
      @Override
      public void onClick(View view) {
        onSettingClick();
      }
    });

    lvSettings = findViewById(R.id.lv_settings);

    if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_BLUETOOTH)) {
      Toast.makeText(this, "bluetooth not supported", Toast.LENGTH_SHORT).show();
    } else {
      btAdapter = BluetoothAdapter.getDefaultAdapter();
      if (!btAdapter.isEnabled()) {
        Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
        int reqCode = 1;
        startActivityForResult(enableBtIntent, reqCode);
//      onActivityResult(reqCode, respCode, enableBtIntent);
      }
      if (!btAdapter.isEnabled()) {
        Log.w(TAG, "bluetooth not on");
      } else {
//          Discovery is not managed by the Activity, but is run as a system service, so an application
//          should always call BluetoothAdapter.cancelDiscovery() even if it did not directly request a discovery, just to be sure.
        btAdapter.cancelDiscovery();
      }
    }

    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .build();


    FaceDetector detector = FaceDetection.getClient(options);

    faceDetector = detector;

    // 异步加载 SD 卡人脸图到内存
    asyncLoadBitmaps2RAM();

//    checkWritePermission();

  }



  private void onAddClick() {
    if (settingPassword.isEmpty()) {
      onAddClickDo();
      return;
    }
    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    LayoutInflater inflater = getLayoutInflater();
    View dialogLayout = inflater.inflate(R.layout.setting_edit_dialog, null);
    TextView tvTitle = dialogLayout.findViewById(R.id.dlg_setting_title);
    TextView tvSubtitle = dialogLayout.findViewById(R.id.dlg_setting_subtitle);
    EditText etInput = dialogLayout.findViewById(R.id.dlg_setting_input);

    tvTitle.setText("需要密码");
    tvSubtitle.setText("当前为门外模式，此操作需授权");
    etInput.setHint("请输入密码");

    builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
      @Override
      public void onClick(DialogInterface dlg, int i) {
        if (settingPassword.equals(etInput.getText().toString())) {
          dlg.dismiss();
          onAddClickDo();
        }
      }
    });
    builder.setView(dialogLayout);
    builder.show();
  }

  private void onAddClickDo() {
    canAddFace = true;
    //Toast.makeText(this, "click", Toast.LENGTH_LONG).show();
    if (!faceDetected) {
      Toast.makeText(this, "未检测到人脸", Toast.LENGTH_SHORT ).show();
    }
  }

  private void onSettingClick() {
    if (settingPassword.isEmpty()) {
      showSettingsDialog();
      return;
    }
    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    LayoutInflater inflater = getLayoutInflater();
    View dialogLayout = inflater.inflate(R.layout.setting_edit_dialog, null);
    TextView tvTitle = dialogLayout.findViewById(R.id.dlg_setting_title);
    TextView tvSubtitle = dialogLayout.findViewById(R.id.dlg_setting_subtitle);
    EditText etInput = dialogLayout.findViewById(R.id.dlg_setting_input);

    tvTitle.setText("需要密码");
    tvSubtitle.setText("当前为门外模式，此操作需授权");
    etInput.setHint("请输入密码");

    builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
      @Override
      public void onClick(DialogInterface dlg, int i) {
        if (settingPassword.equals(etInput.getText().toString())) {
          dlg.dismiss();
          showSettingsDialog();
        }
      }
    });
    builder.setView(dialogLayout);
    builder.show();
  }

  private static final String SETTING_PASSWD_OFF = "关闭门外模式";
  private static final String SETTING_PASSWORD = "启用门外模式";
  private static final String SETTING_DOORNAME = "门的名称";
  private static final String SETTING_TESTOPEN = "测试开门";
  private static final String SETTING_BLEMAC = "蓝牙设备地址";
  private static final String SETTING_FACEREC_THRESHOLD = "人脸识别阈值";
  private static final String SETTING_LIVEBOD_THRESHOLD = "活体检测阈值";
  private static final String SETTING_SERVER_ADDR = "服务器地址";
  private static final String SETTING_DELETE_SOMEONE = "删除某个人";

  private void showSettingsDialog() {
    SettingItem pwdSetting = new SettingItem(SETTING_PASSWORD, "设置密码，添加人脸、进入设置需要", "输入密码，输入空关闭");
    if (!settingPassword.isEmpty()) {
      pwdSetting.setName(SETTING_PASSWD_OFF);
    }
    final SettingItem[] settings = new SettingItem[]{
            pwdSetting,
            new SettingItem(SETTING_DOORNAME, "设置门的名称，以便区分", "输入名称"),
            new SettingItem(SETTING_TESTOPEN, "测试蓝牙、继电器是否工作", "不填"),
            new SettingItem(SETTING_BLEMAC, "设置蓝牙继电器 MAC 地址", BT_MAC_ADDR_HARDCODE),
            new SettingItem(SETTING_FACEREC_THRESHOLD, "设置人脸对比损失函数阈值", "输入 0 以上小数，默认 " + settingFacerecThreshold),
            new SettingItem(SETTING_LIVEBOD_THRESHOLD, "设置活体检测结果阈值", "输入 0-1 之间小数，默认 " + settingLivebodyThreshold),
            new SettingItem(SETTING_SERVER_ADDR, "设置服务器地址", "http://192.168.5.16:9091"),
            new SettingItem(SETTING_DELETE_SOMEONE, "删除某个人已录入照片并停用", "输入录入时名称")
    };

    String[] settingTitles = new String[settings.length];
    for (int i = 0; i < settings.length; i++) {
      settingTitles[i] = settings[i].getName();
    }

    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    LayoutInflater inflater = getLayoutInflater();
    View dialogLayout = inflater.inflate(R.layout.setting_edit_dialog, null);
    TextView tvTitle = dialogLayout.findViewById(R.id.dlg_setting_title);
    TextView tvSubtitle = dialogLayout.findViewById(R.id.dlg_setting_subtitle);
    EditText etInput = dialogLayout.findViewById(R.id.dlg_setting_input);

    AlertDialog dialog = new AlertDialog.Builder(this)
            .setItems(settingTitles, new DialogInterface.OnClickListener() {
              @Override
              public void onClick(DialogInterface dialog, int which) {
                tvTitle.setText(settings[which].getName());
                tvSubtitle.setText(settings[which].getDescription());
                etInput.setHint(settings[which].getHint());

                if (settings[which].getName() == SETTING_TESTOPEN) {
                  // todo 隐藏输入框
                }

                builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
                  @Override
                  public void onClick(DialogInterface dlg, int i) {
                    String input = etInput.getText().toString();
                    switch (settings[which].getName()) {
                      case SETTING_PASSWORD:
                      case SETTING_PASSWD_OFF:
                        settingPassword = input;
                        Log.v(TAG, "setting password to length: " + settingPassword.length());
                        break;
                      case SETTING_FACEREC_THRESHOLD:
                        if (!input.isEmpty()) {
                          settingFacerecThreshold = Float.parseFloat(etInput.getText().toString());
                        }
                        break;
                      case SETTING_LIVEBOD_THRESHOLD:
                        if (!input.isEmpty()) {
                          settingLivebodyThreshold = Float.parseFloat(etInput.getText().toString());
                        }
                        break;
                      case SETTING_TESTOPEN:
                        onTestSwitchClick();
                        break;
                    }
                    dlg.dismiss();
                  }
                });
                builder.setView(dialogLayout);
                builder.show();
              }
            }).create();
    dialog.show();
  }

  private void onTestSwitchClick() {
    blinkBluetoothSwitchAsync();
    Toast.makeText(this, "测试开关", Toast.LENGTH_SHORT).show();
  }

  private void blinkBluetoothSwitchAsync() {
    try {
      btAsyncSendBytes(hexStringToByteArray(BT_CMD_ON));
      Thread.sleep(500);
      btAsyncSendBytes(hexStringToByteArray(BT_CMD_OFF));
    } catch (IOException | InterruptedException e) {
      e.printStackTrace();
    }
  }

  private String byteToHexString(byte[] payload) {
    if (payload == null) return "<empty>";
    StringBuilder stringBuilder = new StringBuilder(payload.length);
    for (byte byteChar : payload)
      stringBuilder.append(String.format("%02X ", byteChar));
    return stringBuilder.toString();
  }

  public static byte[] hexStringToByteArray(String s) {
    int len = s.length();
    byte[] data = new byte[len / 2];
    for (int i = 0; i < len; i += 2) {
      data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4)
              + Character.digit(s.charAt(i+1), 16));
    }
    return data;
  }

  private void asyncLoadBitmaps2RAM() {
    AsyncTask.execute(new Runnable() {
      @Override
      public void run() {
        Log.i(TAG, "开始加载本地人脸");
        int[] countArr = loadLocalBitmaps();
        if (countArr.length >= 2) {
          Log.i(TAG, String.format("完成人脸加载，共 %d 人，取了 %d 张", countArr[0], countArr[1]));
        } else {
          Log.i(TAG, "完成人脸加载");
        }
      }
    });
  }

  private int[] loadLocalBitmaps() {
    if (detector == null) {
      try {
        // FIXME 等待相机、detector 初始化
        Thread.sleep(2000);
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
    }
    String root = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + ImageUtils.APP_DATA_DIR;
    final File myDir = new File(root);
    if (!myDir.exists()) {
      LOGGER.i("directory not found, skip scan");
    }
    int subdirCount = 0;
    int fileCount = 0;
    File[] subdirs = myDir.listFiles(new DirectoryFilter());
    for (int i = 0; i < subdirs.length; i++) {
      LOGGER.d("Checking bitmaps in: %s", subdirs[i].getAbsolutePath());
      if (!subdirs[i].exists()) {
        continue;
      }
      subdirCount++;

      // 过滤 png 文件，再查找 .embedding 文件，找不到则生成并存储
      File[] files = subdirs[i].listFiles(new PngFilter());
      LOGGER.d("Found %d bitmaps in: %s", files.length, subdirs[i].getAbsolutePath());
      // FIXME 返回的文件不保证按名称字母排序
      for (int j = files.length-1; j >= 0; j--) {
        String filepath = files[j].getAbsolutePath();
        LOGGER.i("Loading bitmap and embeddings to RAM, %d, file path: %s", j, filepath);
        String label = subdirs[i].getName();
        SimilarityClassifier.Recognition rec = new SimilarityClassifier.Recognition("0", label, 0.0F, new RectF());
        Bitmap img = BitmapFactory.decodeFile(filepath);

        Object embeddings = null;
        final String embeddingsFilename = filepath + FACE_EMBEDDING_SUFFIX;
        final File embeddingsFile = new File(embeddingsFilename);
        if (embeddingsFile.exists()) {
          embeddings = parseEmbeddingsFile(embeddingsFile);
        } else if (img.getWidth() == TF_OD_API_INPUT_SIZE && img.getHeight() == TF_OD_API_INPUT_SIZE) {
          embeddings = detector.generateEmbeddings(img);
          ImageUtils.saveEmbeddingAsFile(embeddings, subdirs[i].getName(), files[j].getName() + FACE_EMBEDDING_SUFFIX);
        }
        if (embeddings != null) {
          rec.setExtra(embeddings);
          detector.register(label, rec);
          LOGGER.i("Register success %s: %s", label, filepath);
        }
        // 最多取最后的 5 张，为了节约内存
        if (files.length - j > ImageUtils.MAX_IMG_PER_USER) {
          break;
        }
        fileCount++;
      }
    }
    return new int[]{subdirCount, fileCount};
  }

  private Object parseEmbeddingsFile(File f) {
    try {
      FileInputStream in = new FileInputStream(f);
      BufferedReader reader = new BufferedReader(new InputStreamReader(in));

      String content = "";
      String line = "";
      while ((line = reader.readLine()) != null) {
        content += line;
      }
      reader.close();

      Log.d(TAG, "parsing file content to embeddings array: " + content);
      Gson gson = new Gson();
      return gson.fromJson(content, float[][].class);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }

  private void btAsyncSendBytes(byte[] data) throws IOException {
    AsyncTask.execute(new Runnable() {
      @Override
      public void run() {
        try {
          btSendBytes(data);
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    });
  }

  private void btSendBytes(byte[] data) throws IOException {
    if ((btSock == null || !btSock.isConnected() || btWriteError) && !btConnecting) {
      btConnecting = true;
      Log.i(TAG, "try to connect bluetooth...");
      BluetoothDevice btDevice = btAdapter.getRemoteDevice(BT_MAC_ADDR_HARDCODE);
//          Method rfcommMethod = btDevice.getClass().getMethod("createRfcommSocket",
//                  new Class[] { int.class });
//          btSock = (BluetoothSocket) rfcommMethod.invoke(btDevice, Integer.valueOf(1));
      Log.i(TAG, "bluetooth name " + btDevice.getName() + ", mac: " + btDevice.getAddress());

      if (btSock != null && btSock.isConnected()) {
        btSock.close();
      }
      btSock = btDevice.createRfcommSocketToServiceRecord(UUID.fromString(BT_UUID_HARDCODE));
      btSock.connect();
      Log.i(TAG, "connected to bluetooth");
      btConnecting = false;
    }

    if (btSock == null) {
      return;
    }

    OutputStream tmpOut = null;

    Log.i(TAG, "try sending data to bluetooth...");
    try {
      tmpOut = btSock.getOutputStream();
      tmpOut.write(data);
      Log.i(TAG, "sent data to bluetooth..." + byteToHexString(data));
//      byte[] mmBuffer = new byte[1024];
//      Handler handler = new Handler();
//      Message writtenMsg = handler.obtainMessage(MESSAGE_WRITE, -1, -1, mmBuffer);
//      writtenMsg.sendToTarget();
    } catch (IOException e) {
      btWriteError = true;
      Log.e(TAG, "Error occurred when creating output stream", e);
    }

//    for(int i=0; i < data.length; i++){
//      new DataOutputStream(btSock.getOutputStream()).writeByte(data[i]);
//    }
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);


    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      //cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


    int targetW, targetH;
    if (sensorOrientation == 90 || sensorOrientation == 270) {
      targetH = previewWidth;
      targetW = previewHeight;
    }
    else {
      targetW = previewWidth;
      targetH = previewHeight;
    }
    int cropW = (int) (targetW / 2.0);
    int cropH = (int) (targetH / 2.0);

    croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);
    liveInputImg = Bitmap.createBitmap(640, 480, Config.ARGB_8888);
    portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
    faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropW, cropH,
                    sensorOrientation, MAINTAIN_ASPECT);

    frameToLiveTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    LIVE_INPUT_SIZE.getWidth(), LIVE_INPUT_SIZE.getHeight(),
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);


    Matrix frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    targetW, targetH,
                    sensorOrientation, MAINTAIN_ASPECT);



    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }


  @Override
  protected void processImage() {
    ++imageCounter;
    final long currImgCounter = imageCounter;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    LOGGER.v("Preparing image " + currImgCounter + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    final Canvas liveCV = new Canvas(liveInputImg);
    liveCV.drawBitmap(rgbFrameBitmap, frameToLiveTransform, null);

    InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
    faceDetector
            .process(image)
            .addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception e) {

              }
            })
            .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
              @Override
              public void onSuccess(List<Face> faces) {
                if (faces.size() == 0) {
                  updateResults(currImgCounter, new LinkedList<>());
                  LOGGER.v("---- no face detected %s", currImgCounter);
                  canAddFace = false;
                  faceDetected = false;
                  lastRecognizedLabel = "";
                  return;
                }
                LOGGER.v("---- face detected %s", currImgCounter);
                faceDetected = true;
                runInBackground(
                        new Runnable() {
                          @Override
                          public void run() {
                            onFacesDetected(currImgCounter, faces, canAddFace);
                            canAddFace = false;
                          }
                        });
              }

            });


  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }


  // Face Processing
  private Matrix createTransform(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation) {

    Matrix matrix = new Matrix();
    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

    if (applyRotation != 0) {

      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;

  }

  private void showAddFaceDialog(SimilarityClassifier.Recognition rec) {

    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    LayoutInflater inflater = getLayoutInflater();
    View dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null);
    ImageView ivFace = dialogLayout.findViewById(R.id.dlg_image);
    TextView tvTitle = dialogLayout.findViewById(R.id.dlg_title);
    EditText etName = dialogLayout.findViewById(R.id.dlg_input);

    tvTitle.setText("添加人脸");
    if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {
      Matrix matrix = new Matrix();
      matrix.preScale(-1.0f, 1.0f);
      // flip left to right
      ivFace.setImageBitmap(Bitmap.createBitmap(rec.getCrop(), 0, 0, rec.getCrop().getWidth(), rec.getCrop().getHeight(), matrix, true));
    } else {
      ivFace.setImageBitmap(rec.getCrop());
    }
    etName.setHint("如姓名拼音");

    builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
      @Override
      public void onClick(DialogInterface dlg, int i) {

          String label = etName.getText().toString();
          if (label.isEmpty()) {
              return;
          }
          detector.register(label, rec);

          // For examining the actual TF input.
          if (SAVE_FACE_WHEN_ADD) {
            // 存储路径：/sdcard/mp-face-imgs/<label>/<timestamp>.png
            // 仅 112x112 分辨率
            String filename = System.currentTimeMillis() + ".png";
            // 不要用 rec.getCrop() 因为尺寸不符
            ImageUtils.saveBitmap(faceBmp, label, filename);
            ImageUtils.saveEmbeddingAsFile(rec.getExtra(), label,filename + FACE_EMBEDDING_SUFFIX);
          }
          //knownFaces.put(label, rec);

          dlg.dismiss();
      }
    });
    builder.setView(dialogLayout);
    builder.show();

  }


  private void updateResults(long currImgCounter, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

    tracker.trackResults(mappedRecognitions, currImgCounter);
    trackingOverlay.postInvalidate();
    computingDetection = false;
    //adding = false;


    if (mappedRecognitions.size() > 0) {
       LOGGER.i("Adding results");
       SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
       if (rec.getExtra() != null && rec.getCrop() != null) {
         showAddFaceDialog(rec);
       }

    }

    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                showFrameInfo(previewWidth + "x" + previewHeight);
                showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                showInference(lastProcessingTimeMs + "ms");
              }
            });

  }

  private void onFacesDetected(long currImgCounter, List<Face> faces, boolean add) {
    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
    final Canvas canvas = new Canvas(cropCopyBitmap);
    final Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(2.0f);

    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
    switch (MODE) {
      case TF_OD_API:
        minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        break;
    }

    final List<SimilarityClassifier.Recognition> mappedRecognitions =
            new LinkedList<SimilarityClassifier.Recognition>();


    //final List<Classifier.Recognition> results = new ArrayList<>();

    // Note this can be done only once
    int sourceW = rgbFrameBitmap.getWidth();
    int sourceH = rgbFrameBitmap.getHeight();
    int targetW = portraitBmp.getWidth();
    int targetH = portraitBmp.getHeight();
    Matrix transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation);
    final Canvas cv = new Canvas(portraitBmp);

    // draws the original image in portrait mode.
    cv.drawBitmap(rgbFrameBitmap, transform, null);

    final Canvas cvFace = new Canvas(faceBmp);

    boolean saved = false;

    for (Face face : faces) {
      LOGGER.i("FACE: " + face.toString());

      LOGGER.i("Running detection on face " + currImgCounter);
//      results = detector.recognizeImage(croppedBitmap);

      final RectF boundingBox = new RectF(face.getBoundingBox());
//      final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
      final boolean goodConfidence = true; //face.get;
      if (boundingBox != null && goodConfidence) {

        // maps crop coordinates to original
        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
        transform.mapRect(faceBB);

        // translates portrait to origin and scales to fit input inference size
        //cv.drawRect(faceBB, paint);
        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
        Matrix matrix = new Matrix();
        matrix.postTranslate(-faceBB.left, -faceBB.top);
        matrix.postScale(sx, sy);

        cvFace.drawBitmap(portraitBmp, matrix, null);

        //canvas.drawRect(faceBB, paint);

        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

        if (add) {
//          if (face.getSmilingProbability() == null) {
//            Toast.makeText(this, "笑一下呗", Toast.LENGTH_SHORT ).show();
//          }
          LOGGER.d("add face left: %d, right: %d top: %d, bottom: %d, width: %d, height: %d, portrait width: %d, portrait height: %d",
                  (int)faceBB.left, (int)faceBB.right, (int)faceBB.top, (int)faceBB.bottom, (int)faceBB.width(), (int)faceBB.height(), portraitBmp.getWidth(), portraitBmp.getHeight());
          if (faceBB.left + faceBB.width() > portraitBmp.getWidth()) {
            Toast.makeText(this, "脸部在屏幕边缘", Toast.LENGTH_SHORT ).show();
            continue;
          } else {
            // DetectorActivity: face left: 118, right: 506 top: 188, bottom: 646, width: 388, height: 458, portrait width: 480, portrait height: 640
            crop = Bitmap.createBitmap(portraitBmp,
                    (int) faceBB.left,
                    (int) faceBB.top,
                    (int) faceBB.width(),
                    (int) faceBB.height());
          }
        }

        // 活体检测
        LOGGER.i("Running face live check on face " + currImgCounter);
        /**
         *    1       2       3       4        5          6          7            8
         * <p>
         * 888888  888888      88  88      8888888888  88                  88  8888888888
         * 88          88      88  88      88  88      88  88          88  88      88  88
         * 8888      8888    8888  8888    88          8888888888  8888888888          88
         * 88          88      88  88
         * 88          88  888888  888888
         */
        // 原演示 App 使用 640x480 宽高图片 YUV420SP（NV21） 作为输入，方向为 7，输入后会进行人脸检测
        // 这里用 480x640 作为输入，原始方向，使用 MLKit 人脸检测的位置
        // 如果不论真脸假脸，结果 confidence 都在 0.5 附近，检查图片数据格式、人脸位置、方向
        //
        // faceBox 用屏幕左上角作为原点，而 faceBB 用的右上角！
        FaceBox faceBox = new FaceBox((int)(portraitBmp.getWidth()-faceBB.right), (int) faceBB.top, (int) (portraitBmp.getWidth()-faceBB.left), (int) faceBB.bottom, 0F);
//        FaceBox faceBox = NewFaceBoxFrom(faceBB);

        DetectionResult liveDetectResult = liveFaceEngine.detect(getYUVByBitmap(portraitBmp), portraitBmp.getWidth(), portraitBmp.getHeight(), 1, faceBox);
        liveDetectResult.setThreshold(settingLivebodyThreshold);
        boolean realFace =  liveDetectResult.getConfidence() > liveDetectResult.getThreshold();

        LOGGER.i("Face live check result on img %d: confidence: %.4f, threshold: %.4f, real? %b, width: %d, height: %d, face left: %d, top: %d, right: %d, bottom: %d, data size: %d, time: %dms",
                currImgCounter, liveDetectResult.getConfidence(), liveDetectResult.getThreshold(), realFace, portraitBmp.getWidth(), portraitBmp.getHeight(), faceBox.getLeft(), faceBox.getTop(), faceBox.getRight(), faceBox.getBottom(), portraitBmp.getByteCount(), liveDetectResult.getTime());

        // 人脸识别
        final long startTime = SystemClock.uptimeMillis();
        final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, add);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

        if (resultsAux.size() > 0) {

          SimilarityClassifier.Recognition result = resultsAux.get(0);

          extra = result.getExtra();
//          Object extra = result.getExtra();
//          if (extra != null) {
//            LOGGER.i("embeeding retrieved " + extra.toString());
//          }

          float conf = result.getDistance();
          if (conf < settingFacerecThreshold) {
            confidence = conf;
            label = result.getTitle();
            if (result.getId().equals("0")) {
              color = Color.GREEN;
              // lastRecognizedLabel 用来防止同一个人一直开关
              // 人脸移出屏幕、或者换一个人时，才会再开一次

//              if (realFace && lastRecognizedLabel != label) {
//                lastRecognizedLabel = label;
//                Log.i(TAG, label + " 识别成功，开门中...");
//                blinkBluetoothSwitchAsync();
//              }
              if (realFace) {
                Log.i(TAG, label + " 识别成功，开门中...");
                blinkBluetoothSwitchAsync();
              }
            } else {
              color = Color.RED;
            }
          }
          if (!realFace) {
            color = Color.BLACK;
            confidence = liveDetectResult.getConfidence();
            if (label.isEmpty()) {
              label = "假脸！";
            } else {
              label += "，假脸！";
            }
          }
        }

        if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {

          // camera is frontal so the image is flipped horizontally
          // flips horizontally
          Matrix flip = new Matrix();
          if (sensorOrientation == 90 || sensorOrientation == 270) {
            flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          else {
            flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
          flip.mapRect(boundingBox);

        }

        final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                "0", label, confidence, boundingBox);

        result.setColor(color);
        result.setLocation(boundingBox);
        result.setExtra(extra);
        result.setCrop(crop);
        mappedRecognitions.add(result);

      }


    }

    //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }

    updateResults(currImgCounter, mappedRecognitions);


  }
  /*
   * 获取位图的YUV数据
   */
  public static byte[] getYUVByBitmap(Bitmap bitmap) {
    if (bitmap == null) {
      return null;
    }
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();

    int size = width * height;

    int pixels[] = new int[size];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

    // byte[] data = convertColorToByte(pixels);
    byte[] data = rgb2YCbCr420(pixels, width, height);

    return data;
  }

  public static byte[] rgb2YCbCr420(int[] pixels, int width, int height) {
    int len = width * height;
    // yuv格式数组大小，y亮度占len长度，u,v各占len/4长度。
    byte[] yuv = new byte[len * 3 / 2];
    int y, u, v;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // 屏蔽ARGB的透明度值
        int rgb = pixels[i * width + j] & 0x00FFFFFF;
        // 像素的颜色顺序为bgr，移位运算。
        int r = rgb & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = (rgb >> 16) & 0xFF;
        // 套用公式
        y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        // rgb2yuv
        // y = (int) (0.299 * r + 0.587 * g + 0.114 * b);
        // u = (int) (-0.147 * r - 0.289 * g + 0.437 * b);
        // v = (int) (0.615 * r - 0.515 * g - 0.1 * b);
        // RGB转换YCbCr
        // y = (int) (0.299 * r + 0.587 * g + 0.114 * b);
        // u = (int) (-0.1687 * r - 0.3313 * g + 0.5 * b + 128);
        // if (u > 255)
        // u = 255;
        // v = (int) (0.5 * r - 0.4187 * g - 0.0813 * b + 128);
        // if (v > 255)
        // v = 255;
        // 调整
        y = y < 16 ? 16 : (y > 255 ? 255 : y);
        u = u < 0 ? 0 : (u > 255 ? 255 : u);
        v = v < 0 ? 0 : (v > 255 ? 255 : v);
        // 赋值
        yuv[i * width + j] = (byte) y;
        yuv[len + (i >> 1) * width + (j & ~1) + 0] = (byte) u;
        yuv[len + +(i >> 1) * width + (j & ~1) + 1] = (byte) v;
      }
    }
    return yuv;
  }


  //过滤所有以 .png 结尾的文件
  class PngFilter implements FilenameFilter {
    public boolean accept(File dir, String name) {
      return (name.endsWith(".png"));
    }
  }

  //过滤所有目录
  class DirectoryFilter implements FileFilter {
    public boolean accept(File f) {
      return f.isDirectory();
    }
  }
}
