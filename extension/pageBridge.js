/**
 * Deepfake Shield — Main World Page Bridge
 * Runs in the page's execution context (world: "MAIN") so that diagnostic commands
 * like window.deepfakeShieldSelfTest() and window.setDeepfakeShieldDebug()
 * are directly reachable from the normal DevTools console ("top" context).
 */
(function () {
  "use strict";

  if (window.__HV_PAGE_BRIDGE_INITIALIZED__) return;
  window.__HV_PAGE_BRIDGE_INITIALIZED__ = true;

  let reqCounter = 0;

  /**
   * Diagnostic self-test function directly callable from DevTools console.
   * Prints comprehensive status tables and returns diagnostic telemetry.
   */
  window.deepfakeShieldSelfTest = function () {
    return new Promise((resolve) => {
      const reqId = ++reqCounter;
      let handled = false;

      const timeoutId = setTimeout(() => {
        if (handled) return;
        handled = true;
        cleanupListeners();
        console.warn("[Deepfake Shield] Self-test request timed out. Is the extension enabled on this tab?");
        resolve({
          error: "Self-test timed out",
          backendStatus: "TIMEOUT",
          windowFocused: document.hasFocus(),
          activeTiles: 0
        });
      }, 4000);

      function onDataReceived(data) {
        if (handled) return;
        handled = true;
        clearTimeout(timeoutId);
        cleanupListeners();

        data = data || {};
        data.windowFocused = document.hasFocus();

        // Beautiful formatted DevTools console output
        console.group("%c[Deepfake Shield Diagnostics & Self-Test]", "color:#10b981;font-weight:bold;font-size:14px;");

        // 1. Window Focus & Environment State
        const focusColor = data.windowFocused ? "#10b981" : "#f59e0b";
        console.log(`%c• Window Focus State:%c ${data.windowFocused ? "FOCUSED (Meet in foreground)" : "UNFOCUSED (Terminal or another app focused)"}`, "font-weight:bold;", `color:${focusColor};font-weight:bold;`);

        // 2. Backend Health & Connectivity
        console.log(`%c• Backend Server (${data.backendUrl || "http://127.0.0.1:8000"}):%c ${data.backendStatus}`, "font-weight:bold;", "color:#0ea5e9;");
        if (data.healthDetails) {
          console.log("  Health Info:", data.healthDetails);
        }

        // 3. Current Latency & Scheduler Metrics
        if (data.latencyMetrics) {
          console.log(`%c• Performance Metrics:%c RTT: ${data.latencyMetrics.rttMs}ms | Backend: ${data.latencyMetrics.backendLatencyMs}ms | In-Flight: ${data.latencyMetrics.inFlight}`, "font-weight:bold;", "color:#a855f7;");
        }

        // 4. Candidate Videos Audit Table
        console.log(`%c• Discovered Video Elements in DOM:%c ${data.totalVideos || 0}`, "font-weight:bold;", "color:#0ea5e9;");
        if (data.videoAudit && data.videoAudit.length > 0) {
          console.table(data.videoAudit);
        }

        // 5. Active Tracked Tiles & Mirror States
        console.log(`%c• Active Tracked Tiles:%c ${data.activeTiles || 0}`, "font-weight:bold;", "color:#0ea5e9;");
        if (data.tileAudit && data.tileAudit.length > 0) {
          console.table(data.tileAudit);
        }

        // 6. Last Backend Response Sample
        console.log("%c• Last Backend Response Sample:%c", "font-weight:bold;", "color:#0ea5e9;");
        console.dir(data.lastBackendResponse || "No response received yet");

        if (data.lastError) {
          console.warn("[Deepfake Shield] Last error:", data.lastError);
        }

        console.groupEnd();
        resolve(data);
      }

      function messageHandler(event) {
        if (event.data && event.data.type === "HV_SHIELD_SELF_TEST_RES" && event.data.reqId === reqId) {
          onDataReceived(event.data.data);
        }
      }

      function cleanupListeners() {
        window.removeEventListener("message", messageHandler);
      }

      window.addEventListener("message", messageHandler);
      window.postMessage({ type: "HV_SHIELD_SELF_TEST_REQ", reqId, hasFocus: document.hasFocus() }, "*");
    });
  };

  /**
   * Toggles detailed diagnostic logging from DevTools console.
   */
  window.setDeepfakeShieldDebug = function (enabled) {
    const isEnabled = Boolean(enabled);
    window.dispatchEvent(new CustomEvent("HV_SHIELD_SET_DEBUG", { detail: { enabled: isEnabled } }));
    console.log(`[Deepfake Shield] Diagnostic debug logging ${isEnabled ? "ENABLED" : "DISABLED"}`);
    return isEnabled;
  };

  console.log("%c[Deepfake Shield]%c Page bridge active. Run %cdeepfakeShieldSelfTest()%c in console.", "color:#10b981;font-weight:bold;", "color:#94a3b8;", "color:#38bdf8;font-weight:bold;", "color:#94a3b8;");
})();
