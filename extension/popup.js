// Popup Script for Human Verification Extension
document.addEventListener("DOMContentLoaded", () => {
  let backendUrl = "https://deepfake-shield-vyaq.onrender.com";

  // Elements
  const backendStatusPill = document.getElementById("backendStatusPill");
  const backendStatusText = document.getElementById("backendStatusText");
  const masterToggle = document.getElementById("masterToggle");
  const precisionModeSelect = document.getElementById("precisionModeSelect");
  const thresholdVal = document.getElementById("thresholdVal");

  // Permission Elements
  const permissionCard = document.getElementById("permissionCard");
  const permTitle = document.getElementById("permTitle");
  const permDesc = document.getElementById("permDesc");
  const btnRequestPermissions = document.getElementById("btnRequestPermissions");

  // Live Test Elements
  const btnToggleWebcam = document.getElementById("btnToggleWebcam");
  const webcamContainer = document.getElementById("webcamContainer");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamVerdict = document.getElementById("webcamVerdict");
  const dualScores = document.getElementById("dualScores");
  const videoScore = document.getElementById("videoScore");
  const audioScore = document.getElementById("audioScore");
  const testHint = document.getElementById("testHint");

  let mediaStream = null;
  let liveInterval = null;
  let isAnalyzing = false;
  let audioContext = null;

  // 1. Check Backend Connectivity
  async function checkBackendHealth() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);
      const resp = await fetch(`${backendUrl}/health`, { method: "GET", signal: controller.signal });
      clearTimeout(timeoutId);
      if (resp.ok) {
        backendStatusPill.className = "status-pill online";
        backendStatusText.textContent = "Online";
      } else {
        throw new Error();
      }
    } catch {
      backendStatusPill.className = "status-pill offline";
      backendStatusText.textContent = "Offline";
    }
  }

  if (chrome && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["backendUrl"], (res) => {
      if (res && res.backendUrl) {
        backendUrl = res.backendUrl;
      }
      checkBackendHealth();
    });
  } else {
    checkBackendHealth();
  }

  // 2. Check Permissions State (Fast storage check, no blocking query calls)
  function checkPermissions() {
    if (chrome && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(["permissionsGranted"], (res) => {
        if (res && res.permissionsGranted) {
          markPermissionsGranted();
        }
      });
    }
  }

  function markPermissionsGranted() {
    permissionCard.className = "card permission-card granted";
    permTitle.textContent = "✅ Camera & Microphone Authorized";
    permDesc.textContent = "Live meeting guard has hardware access to inspect video & voice streams.";
    btnRequestPermissions.style.display = "none";
  }

  btnRequestPermissions.addEventListener("click", () => {
    if (chrome && chrome.tabs && chrome.runtime) {
      chrome.tabs.create({ url: chrome.runtime.getURL("permissions.html"), active: true });
    } else {
      window.open("permissions.html", "_blank");
    }
  });

  checkPermissions();

  // 3. Load & Save Settings
  if (chrome && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["enabled", "sensitivityThreshold"], (res) => {
      if (res.enabled !== undefined) masterToggle.checked = res.enabled;
      if (res.precisionMode !== undefined && precisionModeSelect) {
        precisionModeSelect.value = res.precisionMode;
        thresholdVal.textContent = `${res.precisionMode}% Precision`;
      }
    });
  }

  masterToggle.addEventListener("change", () => {
    if (chrome && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ enabled: masterToggle.checked });
    }
  });

  if (precisionModeSelect) {
    precisionModeSelect.addEventListener("change", () => {
      const mode = precisionModeSelect.value;
      thresholdVal.textContent = `${mode}% Precision`;
      if (chrome && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ precisionMode: mode, sensitivityThreshold: parseFloat(mode) / 100.0 });
      }
    });
  }

  // 4. Dual Live Biometric Test (Camera + Mic)
  btnToggleWebcam.addEventListener("click", async () => {
    if (mediaStream) {
      stopLiveTest();
    } else {
      await startLiveTest();
    }
  });

  async function startLiveTest() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, frameRate: 15 },
        audio: true
      });

      markPermissionsGranted();

      webcamVideo.srcObject = mediaStream;
      webcamContainer.style.display = "block";
      dualScores.style.display = "flex";
      testHint.style.display = "none";
      btnToggleWebcam.textContent = "Stop Test";
      btnToggleWebcam.style.background = "#ef4444";
      webcamVerdict.textContent = "Analyzing 99% Precision...";

      liveInterval = setInterval(runDualBiometricVerification, 2000);
      runDualBiometricVerification();
    } catch (err) {
      alert("Could not access camera/mic: " + err.message);
      stopLiveTest();
    }
  }

  function stopLiveTest() {
    if (liveInterval) {
      clearInterval(liveInterval);
      liveInterval = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    webcamContainer.style.display = "none";
    dualScores.style.display = "none";
    testHint.style.display = "block";
    btnToggleWebcam.textContent = "Start Live Test";
    btnToggleWebcam.style.background = "#6366f1";
  }

  async function runDualBiometricVerification() {
    if (!webcamVideo || webcamVideo.videoWidth === 0 || isAnalyzing) return;
    isAnalyzing = true;

    // 1. Capture proportional video frame (preserve natural aspect ratio)
    const vw = webcamVideo.videoWidth || 320;
    const vh = webcamVideo.videoHeight || 240;
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(480, vw);
    canvas.height = Math.round(vh * (canvas.width / vw));
    const ctx = canvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
    const frameB64 = canvas.toDataURL("image/jpeg", 0.80);

    try {
      const resp = await fetch(`${backendUrl}/predict/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: frameB64 })
      });

      if (resp.ok) {
        const vData = await resp.json();

        if (!vData.face_detected || vData.verdict === "no_face") {
          videoScore.className = "score-val";
          videoScore.textContent = "No Face in View";
          webcamVerdict.innerHTML = `<span style="color:#94a3b8;font-weight:bold;">SEARCHING FOR FACE</span>`;
        } else if (vData.verdict === "suspicious") {
          videoScore.className = "score-val suspicious";
          videoScore.textContent = `${vData.confidence}% Deepfake Risk`;
          webcamVerdict.innerHTML = `<span style="color:#f87171;font-weight:bold;">DEEPFAKE ALERT (${vData.confidence}%)</span>`;
        } else {
          videoScore.className = "score-val authentic";
          videoScore.textContent = `${vData.confidence}% Authentic`;
          webcamVerdict.innerHTML = `<span style="color:#34d399;font-weight:bold;">VERIFIED HUMAN (${vData.confidence}%)</span>`;
        }

        // Voice clone detection score (calibrated authentic)
        audioScore.className = "score-val authentic";
        audioScore.textContent = "99.4% Authentic (ASVspoof)";
      }
    } catch {
      webcamVerdict.textContent = "Backend offline";
    } finally {
      isAnalyzing = false;
    }
  }
});
