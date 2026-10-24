// Service Worker for Human Verification Deepfake Shield Extension

function sanitizeBackendUrl(url) {
  if (!url || url.includes("localhost") || url.includes("onrender.com") || url.includes("deepfake-shield-backend")) {
    return "http://127.0.0.1:8000";
  }
  return url.replace(/\/+$/, "");
}

// Ensure local PC backend is configured on install and startup
function ensureConfig() {
  chrome.storage.local.get(["backendUrl", "enabled", "scanIntervalMs", "sensitivityThreshold"], (data) => {
    const currentUrl = data ? data.backendUrl : null;
    const cleanUrl = sanitizeBackendUrl(currentUrl);
    chrome.storage.local.set({
      enabled: data && data.enabled !== undefined ? data.enabled : true,
      backendUrl: cleanUrl,
      scanIntervalMs: data && data.scanIntervalMs ? data.scanIntervalMs : 900,
      sensitivityThreshold: data && data.sensitivityThreshold ? data.sensitivityThreshold : 0.88,
      alertNotifications: true,
      lastDetections: []
    });
  });
}

chrome.runtime.onInstalled.addListener(() => {
  ensureConfig();
});

chrome.runtime.onStartup.addListener(() => {
  ensureConfig();
});

// Listen for updates and proxy requests from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "DETECTION_UPDATE") {
    const { verdict, confidence, emotion } = message.payload || {};
    const tabId = sender.tab ? sender.tab.id : undefined;

    if (verdict === "suspicious") {
      chrome.action.setBadgeText({ text: "WARN", tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#ef4444", tabId });
    } else if (verdict === "authentic") {
      chrome.action.setBadgeText({ text: "SAFE", tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#10b981", tabId });
    } else {
      chrome.action.setBadgeText({ text: "", tabId });
    }

    chrome.storage.local.set({
      latestMeetingStatus: {
        verdict: verdict || "authentic",
        confidence: confidence !== undefined ? confidence : 100.0,
        emotion: emotion || "Focused / Calm 👤",
        timestamp: Date.now()
      }
    });

    sendResponse({ status: "acknowledged" });
    return false;
  }

  // Native background fetch proxy: routes frames cleanly to local backend
  if (message.type === "PREDICT_FRAME") {
    const { backendUrl, image_base64, mirrored } = message.payload || {};
    const targetUrl = sanitizeBackendUrl(backendUrl);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    fetch(`${targetUrl}/predict/frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64, mirrored: Boolean(mirrored) }),
      signal: controller.signal
    })
      .then(async (resp) => {
        clearTimeout(timeoutId);
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const data = await resp.json();
        sendResponse({ success: true, data });
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        sendResponse({ success: false, error: err.message });
      });

    return true; // Keep message port open for async response
  }

  // Backend Health check proxy
  if (message.type === "CHECK_HEALTH") {
    const targetUrl = sanitizeBackendUrl(message.payload ? message.payload.backendUrl : null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    fetch(`${targetUrl}/health`, { method: "GET", signal: controller.signal })
      .then(async (resp) => {
        clearTimeout(timeoutId);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        sendResponse({ success: true, data });
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        sendResponse({ success: false, error: err.message });
      });
    return true; // Keep message port open for async response
  }

  return false;
});
