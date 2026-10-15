// Service Worker for Human Verification Deepfake Shield Extension

chrome.runtime.onInstalled.addListener(() => {
  console.log("[Human Verification] Extension installed and active.");
  chrome.storage.local.set({
    enabled: true,
    backendUrl: "http://127.0.0.1:8000",
    scanIntervalMs: 1500,
    sensitivityThreshold: 0.50,
    alertNotifications: true,
    lastDetections: []
  });
});

// Listen for updates from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "DETECTION_UPDATE") {
    const { verdict, confidence, p_fake } = message.payload;
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

    sendResponse({ status: "acknowledged" });
  }
  return true;
});
