# 📱 iOS Native Deepfake Shield (SwiftUI + Apple Neural Engine)

A native iOS application running on-device deepfake inference and involuntary biometric tracking at 60 FPS using Apple CoreML and the Apple Neural Engine (ANE).

---

## 🚀 3-Minute Setup in Xcode

### Prerequisites:
- macOS with Xcode 15+ installed.
- Physical iPhone or iPad (iOS 16+), or Xcode iOS Simulator.

### Step-by-Step Instructions:
1. **Convert or Download the CoreML Model:**
   In your project directory or Colab, export the CoreML model:
   ```bash
   python mobile/ios/export_coreml.py --checkpoint video/outputs/best_model.pt --output mobile/ios/DeepfakeDetector.mlpackage
   ```
2. **Create Xcode Project:**
   - Open Xcode -> **File > New > Project...** -> Select **iOS > App**.
   - Product Name: `DeepfakeMeetingShield`
   - Interface: **SwiftUI**, Language: **Swift**.
3. **Add Project Files:**
   - Drag the following files into your Xcode project:
     - `App.swift`
     - `DeepfakeDetectorView.swift`
     - `DeepfakeDetector.mlpackage` (Ensure "Target Membership" is checked).
   - In **Signing & Capabilities**, select your free Apple ID team.
4. **Build & Run:**
   - Connect your iPhone via USB, select your device in the top bar, and click **Run (⌘R)**.
   - Grant camera permissions when prompted. The AR HUD will overlay real-time biological blink kinematics and deepfake risk scores at 60 FPS!
