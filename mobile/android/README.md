# 🤖 Android Native Deepfake Shield (CameraX + ONNX Mobile NNAPI)

A native Android application running on-device deepfake inference and liveness analysis at 30 FPS using Google CameraX and Microsoft ONNX Runtime Mobile with Android NNAPI hardware acceleration.

---

## 🚀 3-Minute Setup in Android Studio

### Prerequisites:
- Android Studio Iguana / Jellyfish (or newer).
- Physical Android phone (Android 8.0 / API 26+) with USB debugging enabled, or Android Emulator.

### Step-by-Step Instructions:
1. **Export the ONNX Model:**
   In your project directory or Colab, export the ONNX model:
   ```bash
   python video/src/export_onnx.py --checkpoint video/outputs/best_model.pt --output mobile/android/deepfake_detector.onnx
   ```
2. **Create / Open in Android Studio:**
   - In Android Studio, click **New Project > Empty Views Activity**.
   - Package Name: `com.humanverification.shield`
   - Language: **Kotlin**, Build Configuration Language: **Kotlin DSL (`build.gradle.kts`)**.
3. **Copy Files into App Module:**
   - Copy `build.gradle.kts` into your `app/build.gradle.kts`.
   - Copy `AndroidManifest.xml` into `app/src/main/AndroidManifest.xml`.
   - Copy `MainActivity.kt` and `DeepfakeAnalyzer.kt` into `app/src/main/java/com/humanverification/shield/`.
   - Copy `deepfake_detector.onnx` into `app/src/main/assets/deepfake_detector.onnx`.
4. **Build & Run:**
   - Click **Sync Project with Gradle Files**.
   - Click **Run 'app' (Shift+F10)**.
   - Point your phone's front camera at your face: the HUD will instantly report `AUTHENTIC` or `SUSPICIOUS` with millisecond inference latency!
