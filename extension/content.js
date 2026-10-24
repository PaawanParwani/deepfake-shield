// Human Verification — Real-Time Google Meet & Video Call Shield
(function () {
  "use strict";

  // 1. NEVER inject into sub-frames or iframes
  if (window !== window.top) return;
  if (window.__HV_SHIELD_INJECTED__) return;
  window.__HV_SHIELD_INJECTED__ = true;

  let config = {
    enabled: true,
    backendUrl: "http://127.0.0.1:8000",
    scanIntervalMs: 500,
    scanIntervalLockedMs: 300,
    sensitivityThreshold: 0.88,
    mirrorMode: "auto", // "auto" (default: CSS matrix detection), "flipped" (manual override), "normal" (manual override)
    debug: true
  };

  // State
  let inFlight = false;
  let activeVideo = null;
  let activeVideoResizeObserver = null;
  let activeVideoMutationObserver = null;
  let hudBadge = null;
  let overlayRoot = null;
  let dockPill = null;
  let consecutiveNoFaceCount = 0;
  let consecutiveBackendFailures = 0;
  let loopRunning = false;
  let lastCapturedW = 320;
  let lastCapturedH = 180;

  // Active Multi-Face Tracking Map: id -> TrackedFace
  const trackedFaces = new Map();
  let faceIdCounter = 0;

  function isDebugEnabled() {
    return (typeof window !== "undefined" && window.__HV_DEBUG__ !== undefined)
      ? Boolean(window.__HV_DEBUG__)
      : Boolean(config.debug);
  }

  // Developer console convenience: window.setDeepfakeShieldDebug(true/false)
  if (typeof window !== "undefined") {
    window.setDeepfakeShieldDebug = function (enabled) {
      config.debug = Boolean(enabled);
      window.__HV_DEBUG__ = config.debug;
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ debug: config.debug });
      }
      console.log(`[Deepfake Shield] Debug logging set to: ${config.debug}`);
    };
  }

  function debugLog(...args) {
    if (isDebugEnabled()) {
      console.log("%c[Deepfake Shield]", "color: #0ea5e9; font-weight: bold;", ...args);
    }
  }

  function escapeHTML(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Sync extension storage config (persists user preferences across reconnects)
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(
      ["enabled", "backendUrl", "scanIntervalMs", "scanIntervalLockedMs", "sensitivityThreshold", "mirrorMode", "debug"],
      (stored) => {
        if (stored) {
          if (!stored.backendUrl || stored.backendUrl.includes("localhost") || stored.backendUrl.includes("onrender.com") || stored.backendUrl.includes("deepfake-shield-backend")) {
            stored.backendUrl = "http://127.0.0.1:8000";
            chrome.storage.local.set({ backendUrl: "http://127.0.0.1:8000" });
          }
          if (stored.mirrorMode === undefined || stored.mirrorMode === null) {
            stored.mirrorMode = "auto";
          }
          config = { ...config, ...stored };
        }
        updateDockPill();
        updateFlipButtonLabel();
        repositionTrackedFaces();
      }
    );

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local") {
        for (let k in changes) {
          if (k === "backendUrl") {
            const v = changes[k].newValue;
            config.backendUrl = (!v || v.includes("localhost") || v.includes("onrender.com")) ? "http://127.0.0.1:8000" : v;
          } else if (k in config) {
            config[k] = changes[k].newValue;
          }
        }
        updateDockPill();
        updateFlipButtonLabel();
        repositionTrackedFaces();
      }
    });
  }

  // Fixed Viewport Overlay Root (Immune to parent overflow / transforms)
  function getOrCreateOverlayRoot() {
    if (overlayRoot && document.body.contains(overlayRoot)) return overlayRoot;
    let root = document.getElementById("hv-overlay-root");
    if (!root) {
      root = document.createElement("div");
      root.id = "hv-overlay-root";
      document.body.appendChild(root);
    }
    overlayRoot = root;
    return overlayRoot;
  }

  // Lightweight 320p Compact Frame Capture (~15KB, zero system lag)
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  function captureCompactFrame(video) {
    if (!video || video.paused || video.ended || video.readyState < 2) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || vw < 40 || vh < 40) return null;

    try {
      const maxDim = 320;
      const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
      const targetW = Math.max(160, Math.round(vw * scale));
      const targetH = Math.max(120, Math.round(vh * scale));

      if (offscreenCanvas.width !== targetW || offscreenCanvas.height !== targetH) {
        offscreenCanvas.width = targetW;
        offscreenCanvas.height = targetH;
      }

      offscreenCtx.imageSmoothingEnabled = true;
      offscreenCtx.imageSmoothingQuality = "medium";
      offscreenCtx.drawImage(video, 0, 0, targetW, targetH);

      lastCapturedW = targetW;
      lastCapturedH = targetH;

      return {
        dataUrl: offscreenCanvas.toDataURL("image/jpeg", 0.75),
        capturedW: targetW,
        capturedH: targetH
      };
    } catch (err) {
      return null;
    }
  }

  // Parse CSS transform string into 2x2 affine matrix [a, b, c, d]
  // In CSS matrix(a, b, c, d, tx, ty):
  //   [ a  c  tx ]
  //   [ b  d  ty ]
  // In CSS matrix3d(m11, m12, ... m44):
  //   [ m11  m21  ... ]
  //   [ m12  m22  ... ]
  function parseTransform2D(transformStr) {
    if (!transformStr || transformStr === "none") {
      return [1, 0, 0, 1]; // Identity [a, b, c, d]
    }
    const t = transformStr.trim().toLowerCase();
    if (t.startsWith("matrix(")) {
      const parts = t.slice(7, -1).split(",").map((s) => parseFloat(s.trim()));
      if (parts.length >= 4 && !parts.slice(0, 4).some(isNaN)) {
        return [parts[0], parts[1], parts[2], parts[3]];
      }
    } else if (t.startsWith("matrix3d(")) {
      const parts = t.slice(9, -1).split(",").map((s) => parseFloat(s.trim()));
      // Column-major order:
      // a = a1 (parts[0]), b = b1 (parts[1]), c = a2 (parts[4]), d = b2 (parts[5])
      if (parts.length >= 16 && !isNaN(parts[0]) && !isNaN(parts[1]) && !isNaN(parts[4]) && !isNaN(parts[5])) {
        return [parts[0], parts[1], parts[4], parts[5]];
      }
    }
    return [1, 0, 0, 1];
  }

  // Multiply two 2x2 matrices: M1 x M2
  // [a1, c1] x [a2, c2] = [a1*a2 + c1*b2, a1*c2 + c1*d2]
  // [b1, d1]   [b2, d2]   [b1*a2 + d1*b2, b1*c2 + d1*d2]
  function multiply2x2(m1, m2) {
    const [a1, b1, c1, d1] = m1;
    const [a2, b2, c2, d2] = m2;
    return [
      a1 * a2 + c1 * b2,
      b1 * a2 + d1 * b2,
      a1 * c2 + c1 * d2,
      b1 * c2 + d1 * d2
    ];
  }

  // Walk from target <video> up through ancestors, reading computed transforms,
  // multiplying matrices from video to root, and detecting negative x-scale (a < 0).
  function detectVideoMirroring(video) {
    if (!video) return { isMirrored: false, scaleX: 1, transforms: [] };

    let el = video;
    const transforms = [];
    const chain = [];

    while (el && el !== document.documentElement) {
      try {
        const compStyle = window.getComputedStyle(el);
        const t = (compStyle.transform || compStyle.webkitTransform || "").trim();
        if (t && t !== "none") {
          const identifier = `${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}${el.className ? '.' + String(el.className).trim().split(/\s+/)[0] : ''}`;
          transforms.push(`${identifier}: ${t}`);
          chain.push(parseTransform2D(t));
        }
      } catch {}
      el = el.parentElement;
    }

    // Cumulative transform: M_total = T_root x ... x T_parent x T_video
    // chain is [T_video, T_parent, T_grandparent, ...]
    let cumulative = [1, 0, 0, 1];
    for (let i = chain.length - 1; i >= 0; i--) {
      cumulative = multiply2x2(cumulative, chain[i]);
    }

    const scaleX = cumulative[0];
    const isMirrored = scaleX < -0.001;

    return {
      isMirrored,
      scaleX,
      cumulative,
      transforms
    };
  }

  function resolveMirrorState(video) {
    const mirrorInfo = detectVideoMirroring(video);
    let shouldFlip = false;
    let modeLabel = config.mirrorMode || "auto";

    if (config.mirrorMode === "flipped") {
      shouldFlip = true;
      modeLabel = "forced flipped";
    } else if (config.mirrorMode === "normal") {
      shouldFlip = false;
      modeLabel = "forced normal";
    } else {
      // "auto" (default): Follow computed CSS matrix transform
      shouldFlip = mirrorInfo.isMirrored;
      modeLabel = "auto";
    }

    return {
      shouldFlip,
      modeLabel,
      mirrorInfo
    };
  }

  // Mathematical CSS object-fit Geometry (Letterbox / Pillarbox / Crop Compensated)
  function getRenderedVideoGeometry(video) {
    if (!video || !document.body.contains(video)) return null;

    const rect = video.getBoundingClientRect();
    const elemW = rect.width;
    const elemH = rect.height;
    const vw = video.videoWidth || elemW || 640;
    const vh = video.videoHeight || elemH || 480;

    if (elemW <= 0 || elemH <= 0 || vw <= 0 || vh <= 0) return null;

    const style = window.getComputedStyle(video);
    const objectFit = style.objectFit || "cover";

    let renderW = elemW;
    let renderH = elemH;
    let offsetX = 0;
    let offsetY = 0;

    const videoRatio = vw / vh;
    const elemRatio = elemW / elemH;

    if (objectFit === "contain") {
      if (elemRatio > videoRatio) {
        // Pillarboxed: black bars left and right
        renderH = elemH;
        renderW = elemH * videoRatio;
        offsetX = (elemW - renderW) / 2;
        offsetY = 0;
      } else {
        // Letterboxed: black bars top and bottom
        renderW = elemW;
        renderH = elemW / videoRatio;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      }
    } else {
      // object-fit: cover (standard for Google Meet participant video tiles)
      if (elemRatio > videoRatio) {
        // Top and bottom cropped
        renderW = elemW;
        renderH = elemW / videoRatio;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      } else {
        // Left and right cropped
        renderH = elemH;
        renderW = elemH * videoRatio;
        offsetX = (elemW - renderW) / 2;
        offsetY = 0;
      }
    }

    return {
      left: rect.left,
      top: rect.top,
      width: elemW,
      height: elemH,
      renderW,
      renderH,
      offsetX,
      offsetY,
      objectFit,
      vw,
      vh,
      dpr: window.devicePixelRatio || 1
    };
  }

  function isElementVisible(el) {
    if (!el || !document.body.contains(el)) return false;
    try {
      const rect = el.getBoundingClientRect();
      if (rect.width < 80 || rect.height < 60) return false;
      if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) return false;
      return true;
    } catch {
      return false;
    }
  }

  function getCandidateVideos() {
    const videos = Array.from(document.querySelectorAll("video"));
    return videos.filter((v) => {
      if (v.paused || v.ended || v.readyState < 2) return false;
      if (!v.videoWidth || !v.videoHeight || v.videoWidth < 40 || v.videoHeight < 40) return false;
      return isElementVisible(v);
    });
  }

  // Deterministically select the primary video element
  function findPrimaryVideo() {
    const candidates = getCandidateVideos();
    if (candidates.length === 0) return null;

    // Prefer active self-view video if available
    const selfVideo = candidates.find((v) => {
      return !!(
        v.closest("[data-self-name]") ||
        v.closest("[data-is-self-view='true']") ||
        v.getAttribute("data-is-self-view") === "true" ||
        v.className.includes("self") ||
        (v.parentElement && v.parentElement.className.includes("self"))
      );
    });
    if (selfVideo) return selfVideo;

    // Otherwise select the largest visible active video
    let best = candidates[0];
    let maxArea = 0;
    for (const v of candidates) {
      const r = v.getBoundingClientRect();
      const area = r.width * r.height;
      if (area > maxArea) {
        maxArea = area;
        best = v;
      }
    }

    if (activeVideo && candidates.includes(activeVideo)) {
      const curRect = activeVideo.getBoundingClientRect();
      if (curRect.width * curRect.height >= maxArea * 0.70) {
        return activeVideo;
      }
    }

    return best;
  }

  // Real-time ResizeObserver: Re-anchors bounding boxes instantly during layout changes / window resize
  function attachVideoObservers(video) {
    if (activeVideoResizeObserver) {
      activeVideoResizeObserver.disconnect();
      activeVideoResizeObserver = null;
    }

    if (window.ResizeObserver && video) {
      activeVideoResizeObserver = new ResizeObserver(() => {
        repositionTrackedFaces();
        updateHUDBadgePosition();
      });
      activeVideoResizeObserver.observe(video);
      if (video.parentElement) {
        activeVideoResizeObserver.observe(video.parentElement);
      }
    }
  }

  // Repositions currently tracked faces immediately using latest geometry & transform
  function repositionTrackedFaces() {
    if (!activeVideo || trackedFaces.size === 0) return;
    const geom = getRenderedVideoGeometry(activeVideo);
    if (!geom) return;

    const { shouldFlip } = resolveMirrorState(activeVideo);
    updateFlipButtonLabel();

    for (const [id, face] of trackedFaces.entries()) {
      if (!face.lastNormBox || !face.overlayElement) continue;

      let [x_norm, y_norm, w_norm, h_norm] = face.lastNormBox;
      // If mirrored, flip the whole box: x_screen = 1 - (x_norm + w_norm)
      const screen_x_norm = shouldFlip ? (1.0 - (x_norm + w_norm)) : x_norm;

      const targetX = geom.left + geom.offsetX + (screen_x_norm * geom.renderW);
      const targetY = geom.top + geom.offsetY + (y_norm * geom.renderH);
      const targetW = w_norm * geom.renderW;
      const targetH = h_norm * geom.renderH;

      face.smoothed = { x: targetX, y: targetY, w: targetW, h: targetH };

      const el = face.overlayElement;
      el.style.left = `${Math.round(targetX)}px`;
      el.style.top = `${Math.round(targetY)}px`;
      el.style.width = `${Math.round(targetW)}px`;
      el.style.height = `${Math.round(targetH)}px`;
    }
  }

  // Single-Thread Non-Overlapping Live Frame Scan (Promise-based)
  function executeLiveScan() {
    return new Promise((resolve) => {
      if (!config.enabled || inFlight) {
        resolve();
        return;
      }

      const video = findPrimaryVideo();
      if (!video) {
        hideHUD();
        resolve();
        return;
      }

      if (activeVideo !== video) {
        activeVideo = video;
        attachHUD(video);
        attachVideoObservers(video);
      }

      const capture = captureCompactFrame(video);
      if (!capture) {
        resolve();
        return;
      }

      inFlight = true;
      const reqStart = performance.now();

      // Watchdog timeout to prevent frozen requests
      const watchdogId = setTimeout(() => {
        if (inFlight) {
          inFlight = false;
          consecutiveBackendFailures++;
          if (consecutiveBackendFailures >= 6) {
            renderOffline();
          }
          resolve();
        }
      }, 7500);

      const resolved = resolveMirrorState(video);

      if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
        try {
          chrome.runtime.sendMessage(
            {
              type: "PREDICT_FRAME",
              payload: {
                backendUrl: config.backendUrl,
                image_base64: capture.dataUrl,
                mirrored: resolved.shouldFlip
              }
            },
            (response) => {
              clearTimeout(watchdogId);
              inFlight = false;

              if (chrome.runtime.lastError || !response || !response.success) {
                consecutiveBackendFailures++;
                if (consecutiveBackendFailures >= 6) {
                  renderOffline();
                }
                resolve();
                return;
              }

              consecutiveBackendFailures = 0;
              const clientLatency = Math.round(performance.now() - reqStart);
              try {
                processDetections(response.data, video, capture, clientLatency);
              } catch (err) {
                debugLog("Detection process error:", err);
              }
              resolve();
            }
          );
          return;
        } catch (sendErr) {
          clearTimeout(watchdogId);
          inFlight = false;
          consecutiveBackendFailures++;
          if (consecutiveBackendFailures >= 6) {
            renderOffline();
          }
          resolve();
          return;
        }
      }

      inFlight = false;
      resolve();
    });
  }

  // Continuous Sequential Scanning Loop (Zero Race Conditions, Zero IPC Flood)
  async function startDetectionLoop() {
    if (loopRunning) return;
    loopRunning = true;

    while (config.enabled) {
      if (document.hidden) {
        await sleep(1000);
        continue;
      }

      try {
        await executeLiveScan();
      } catch (err) {
        debugLog("Detection scan loop error:", err);
      }

      const isLocked = trackedFaces.size > 0;
      const interval = isLocked ? config.scanIntervalLockedMs : config.scanIntervalMs;
      await sleep(interval);
    }

    loopRunning = false;
  }

  function updateFlipButtonLabel() {
    if (!hudBadge) return;
    const flipBtn = hudBadge.querySelector("#hv-btn-flip");
    if (!flipBtn) return;

    if (config.mirrorMode === "auto") {
      const mirrorInfo = detectVideoMirroring(activeVideo);
      flipBtn.textContent = mirrorInfo.isMirrored ? "⇄ Auto (Mirrored)" : "⇄ Auto (Normal)";
      flipBtn.title = "Mirror Mode: Auto (detecting CSS transforms). Click to cycle: Auto -> Force Flipped -> Force Normal.";
      flipBtn.style.borderColor = "rgba(255, 255, 255, 0.35)";
    } else if (config.mirrorMode === "flipped") {
      flipBtn.textContent = "⇄ Force Flipped";
      flipBtn.title = "Mirror Mode: Forced Flipped (Manual Override). Click to cycle: Auto -> Force Flipped -> Force Normal.";
      flipBtn.style.borderColor = "rgba(245, 158, 11, 0.85)";
    } else {
      flipBtn.textContent = "⇄ Force Normal";
      flipBtn.title = "Mirror Mode: Forced Normal (Manual Override). Click to cycle: Auto -> Force Flipped -> Force Normal.";
      flipBtn.style.borderColor = "rgba(59, 130, 246, 0.85)";
    }
  }

  function attachHUD(video) {
    getOrCreateOverlayRoot();

    if (!hudBadge) {
      hudBadge = document.createElement("div");
      hudBadge.className = "hv-hud-badge hv-authentic";
      hudBadge.innerHTML = `
        <div class="hv-pulse-dot"></div>
        <div class="hv-hud-text">
          <span id="hv-badge-title">PROTECTED</span>
          <span class="hv-hud-score" id="hv-badge-score">Scanning face...</span>
        </div>
        <button id="hv-btn-flip" title="Toggle Mirror / Flip (⇄)" class="hv-badge-flip-btn">⇄ Auto</button>
      `;
      document.body.appendChild(hudBadge);

      const flipBtn = hudBadge.querySelector("#hv-btn-flip");
      if (flipBtn) {
        updateFlipButtonLabel();
        flipBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          // Cycle: auto -> flipped -> normal -> auto
          if (config.mirrorMode === "auto") {
            config.mirrorMode = "flipped";
          } else if (config.mirrorMode === "flipped") {
            config.mirrorMode = "normal";
          } else {
            config.mirrorMode = "auto";
          }
          if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({ mirrorMode: config.mirrorMode });
          }
          updateFlipButtonLabel();
          repositionTrackedFaces();
          executeLiveScan();
        });
      }
    }

    const candidates = getCandidateVideos();
    candidates.forEach((v) => {
      if (!v.__hv_click_bound) {
        v.__hv_click_bound = true;
        v.addEventListener("click", () => {
          activeVideo = v;
          consecutiveNoFaceCount = 0;
          attachHUD(v);
          attachVideoObservers(v);
          executeLiveScan();
        });
      }
    });
  }

  function updateHUDBadgePosition() {
    if (!hudBadge || !activeVideo || !document.body.contains(activeVideo)) return;
    const rect = activeVideo.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      hudBadge.style.display = "none";
      return;
    }
    hudBadge.style.display = "flex";
    hudBadge.style.left = `${Math.round(rect.left + 14)}px`;
    hudBadge.style.top = `${Math.round(rect.top + 14)}px`;
  }

  // Multi-Face Tracking, Normalized Coordinate Mapping & Diagnostic Telemetry
  function processDetections(data, video, capture, clientLatency) {
    if (!hudBadge || !video) return;
    const root = getOrCreateOverlayRoot();

    const { verdict, face_detected, latency_ms, emotion } = data;
    const geom = getRenderedVideoGeometry(video);
    if (!geom) return;

    const fw = data.frame_width || capture.capturedW || 320;
    const fh = data.frame_height || capture.capturedH || 180;

    // Automatic CSS matrix mirroring resolution & manual override
    const resolved = resolveMirrorState(video);
    const shouldFlip = resolved.shouldFlip;
    const modeLabel = resolved.modeLabel;
    const transformInfo = resolved.mirrorInfo;
    updateFlipButtonLabel();

    // ZERO FACES DETECTED: Instant Ghost Box Purge
    if (!face_detected || verdict === "no_face" || !data.faces || data.faces.length === 0) {
      consecutiveNoFaceCount++;
      hudBadge.className = "hv-hud-badge hv-no-face";
      hudBadge.querySelector("#hv-badge-title").textContent = "NO FACE DETECTED";
      hudBadge.querySelector("#hv-badge-score").textContent = "Position face in camera";
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = "Scanning for participant face...";
      }

      for (const [id, face] of trackedFaces.entries()) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
      }
      trackedFaces.clear();
      updateHUDBadgePosition();
      return;
    }

    consecutiveNoFaceCount = 0;
    const detections = data.faces;
    const matchedTrackIds = new Set();

    const mappedDetections = detections.map((det, idx) => {
      const box = det.bounding_box || [0, 0, 0, 0];

      // Extract normalized bounding box from backend [x_norm, y_norm, w_norm, h_norm]
      let x_norm, y_norm, w_norm, h_norm;
      if (det.norm_xywh && det.norm_xywh.length >= 4) {
        x_norm = det.norm_xywh[0];
        y_norm = det.norm_xywh[1];
        w_norm = det.norm_xywh[2];
        h_norm = det.norm_xywh[3];
      } else if (det.norm_bbox && det.norm_bbox.length >= 4) {
        x_norm = det.norm_bbox[0];
        y_norm = det.norm_bbox[1];
        w_norm = det.norm_bbox[2] - det.norm_bbox[0];
        h_norm = det.norm_bbox[3] - det.norm_bbox[1];
      } else {
        x_norm = (box[0] || 0) / fw;
        y_norm = (box[1] || 0) / fh;
        w_norm = ((box[2] || 0) - (box[0] || 0)) / fw;
        h_norm = ((box[3] || 0) - (box[1] || 0)) / fh;
      }

      x_norm = Math.max(0.0, Math.min(1.0, x_norm));
      y_norm = Math.max(0.0, Math.min(1.0, y_norm));
      w_norm = Math.max(0.0, Math.min(1.0 - x_norm, w_norm));
      h_norm = Math.max(0.0, Math.min(1.0 - y_norm, h_norm));

      const rawNormBox = [x_norm, y_norm, w_norm, h_norm];

      // Exact horizontal mirror mapping: x_screen = 1 - (x_norm + w_norm)
      const screen_x_norm = shouldFlip ? (1.0 - (x_norm + w_norm)) : x_norm;

      const targetX = geom.left + geom.offsetX + (screen_x_norm * geom.renderW);
      const targetY = geom.top + geom.offsetY + (y_norm * geom.renderH);
      const targetW = w_norm * geom.renderW;
      const targetH = h_norm * geom.renderH;
      const centerX = targetX + targetW / 2;
      const centerY = targetY + targetH / 2;

      // Real-Time Engineering Diagnostic Log (gated behind debug flag)
      if (isDebugEnabled() && idx === 0) {
        console.log(
          `%c[Deepfake Shield Per-Frame Debug]%c\n` +
          `• backend bbox: [${box.join(', ')}]\n` +
          `• norm_xywh: [${x_norm.toFixed(3)}, ${y_norm.toFixed(3)}, ${w_norm.toFixed(3)}, ${h_norm.toFixed(3)}]\n` +
          `• frame sent: ${capture.capturedW}x${capture.capturedH}\n` +
          `• video: ${video.videoWidth}x${video.videoHeight} (ratio: ${(video.videoWidth / video.videoHeight).toFixed(3)})\n` +
          `• elem: ${Math.round(geom.width)}x${Math.round(geom.height)} (client: ${video.clientWidth}x${video.clientHeight})\n` +
          `• getBoundingClientRect: left=${Math.round(geom.left)}, top=${Math.round(geom.top)}, w=${Math.round(geom.width)}, h=${Math.round(geom.height)}\n` +
          `• object-fit: ${geom.objectFit}, renderW: ${Math.round(geom.renderW)}, renderH: ${Math.round(geom.renderH)}, offset: (${Math.round(geom.offsetX)}, ${Math.round(geom.offsetY)})\n` +
          `• transforms: [${transformInfo.transforms.join(' | ') || 'none'}]\n` +
          `• mirror applied: ${shouldFlip} (${modeLabel})\n` +
          `• screen target: left=${Math.round(targetX)}px, top=${Math.round(targetY)}px, w=${Math.round(targetW)}px, h=${Math.round(targetH)}px\n` +
          `• devicePixelRatio: ${window.devicePixelRatio}`,
          "color: #0ea5e9; font-weight: bold;",
          "color: inherit;"
        );
      }

      return {
        det,
        targetIndex: idx + 1,
        targetX,
        targetY,
        targetW,
        targetH,
        centerX,
        centerY,
        rawNormBox
      };
    });

    mappedDetections.forEach((mDet) => {
      let bestMatch = null;
      if (trackedFaces.size >= 1) {
        bestMatch = Array.from(trackedFaces.values())[0];
      }

      if (!bestMatch) {
        faceIdCounter++;
        const newId = `track_${faceIdCounter}`;

        const overlayEl = document.createElement("div");
        overlayEl.className = "hv-face-overlay";
        overlayEl.style.opacity = "0";

        const tagEl = document.createElement("div");
        tagEl.className = "hv-face-tag";
        overlayEl.appendChild(tagEl);

        const crosshairEl = document.createElement("div");
        crosshairEl.className = "hv-crosshair";
        overlayEl.appendChild(crosshairEl);

        root.appendChild(overlayEl);

        bestMatch = {
          id: newId,
          label: `Face #${mDet.targetIndex}`,
          overlayElement: overlayEl,
          tagElement: tagEl,
          crosshairElement: crosshairEl,
          smoothed: { x: mDet.targetX, y: mDet.targetY, w: mDet.targetW, h: mDet.targetH }
        };

        trackedFaces.set(newId, bestMatch);
      }

      matchedTrackIds.add(bestMatch.id);
      bestMatch.lastNormBox = mDet.rawNormBox;
      bestMatch.emotion = mDet.det.emotion || "Focused / Calm 👤";
      bestMatch.verdict = mDet.det.verdict || "authentic";
      bestMatch.confidence = mDet.det.confidence || 100.0;

      // Measure displacement of the face center relative to rendered video dimensions
      const prevCenterX = bestMatch.smoothed.x + bestMatch.smoothed.w / 2;
      const prevCenterY = bestMatch.smoothed.y + bestMatch.smoothed.h / 2;
      const relDx = Math.abs(mDet.centerX - prevCenterX) / Math.max(1, geom.renderW);
      const relDy = Math.abs(mDet.centerY - prevCenterY) / Math.max(1, geom.renderH);
      const relDist = Math.sqrt(relDx * relDx + relDy * relDy);

      // Dynamic EMA smoothing: snap immediately when moving >10% of frame, smooth jitter during subtle motion
      const isFastMovement = relDist > 0.10;
      const alpha = isFastMovement ? 1.0 : 0.70;

      bestMatch.smoothed.x = alpha * mDet.targetX + (1 - alpha) * bestMatch.smoothed.x;
      bestMatch.smoothed.y = alpha * mDet.targetY + (1 - alpha) * bestMatch.smoothed.y;
      bestMatch.smoothed.w = alpha * mDet.targetW + (1 - alpha) * bestMatch.smoothed.w;
      bestMatch.smoothed.h = alpha * mDet.targetH + (1 - alpha) * bestMatch.smoothed.h;

      const el = bestMatch.overlayElement;
      if (el) {
        // Disable CSS transition during fast head movements to eliminate visual lag/trailing
        el.style.transition = isFastMovement
          ? "none"
          : "left 0.08s ease-out, top 0.08s ease-out, width 0.08s ease-out, height 0.08s ease-out, border-color 0.25s, box-shadow 0.25s, opacity 0.25s";
        el.style.display = "block";
        el.style.left = `${Math.round(bestMatch.smoothed.x)}px`;
        el.style.top = `${Math.round(bestMatch.smoothed.y)}px`;
        el.style.width = `${Math.round(bestMatch.smoothed.w)}px`;
        el.style.height = `${Math.round(bestMatch.smoothed.h)}px`;
        el.style.opacity = "1";

        const nameDisplay = bestMatch.label || "Face";
        const emotionDisplay = bestMatch.emotion;

        if (bestMatch.verdict === "suspicious") {
          el.className = "hv-face-overlay hv-suspicious-box";
          if (bestMatch.tagElement) {
            bestMatch.tagElement.innerHTML = `
              <span class="hv-tag-identity">${escapeHTML(nameDisplay)}</span>
              <span class="hv-tag-verdict hv-v-susp">${bestMatch.confidence}% Risk (Deepfake)</span>
              <span class="hv-tag-emotion">${escapeHTML(emotionDisplay)}</span>
            `;
          }
        } else {
          el.className = "hv-face-overlay";
          if (bestMatch.tagElement) {
            bestMatch.tagElement.innerHTML = `
              <span class="hv-tag-identity">${escapeHTML(nameDisplay)}</span>
              <span class="hv-tag-verdict hv-v-auth">${bestMatch.confidence}% Real</span>
              <span class="hv-tag-emotion">${escapeHTML(emotionDisplay)}</span>
            `;
          }
        }
      }
    });

    // Remove any unmatched faces
    for (const [id, face] of trackedFaces.entries()) {
      if (!matchedTrackIds.has(id)) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
        trackedFaces.delete(id);
      }
    }

    updateHUDBadgePosition();

    // Floating Top HUD Badge
    const activeList = Array.from(trackedFaces.values());
    const totalFaces = activeList.length;
    const suspiciousFaces = activeList.filter((f) => f.verdict === "suspicious");
    const primaryEmotion = (activeList.length > 0 && activeList[0].emotion) || emotion || "Focused / Calm 👤";

    if (suspiciousFaces.length > 0) {
      const maxRisk = Math.max(...suspiciousFaces.map((f) => f.confidence));
      hudBadge.className = "hv-hud-badge hv-suspicious";
      hudBadge.querySelector("#hv-badge-title").textContent = totalFaces > 1 ? `DEEPFAKE ALERT (${suspiciousFaces.length}/${totalFaces})` : "DEEPFAKE DETECTED";
      hudBadge.querySelector("#hv-badge-score").textContent = `${maxRisk}% Risk (${latency_ms || clientLatency}ms)`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = `⚠️ Deepfake Alert (${maxRisk}% Risk)`;
      }
    } else {
      const minAuthConf = totalFaces > 0 ? Math.min(...activeList.map((f) => f.confidence)) : 100.0;
      hudBadge.className = "hv-hud-badge hv-authentic";
      hudBadge.querySelector("#hv-badge-title").textContent = "PARTICIPANT • VERIFIED";
      hudBadge.querySelector("#hv-badge-score").textContent = `${minAuthConf}% Authentic • ${primaryEmotion}`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = `Participant Protected • ${primaryEmotion}`;
      }
    }

    broadcastStatusUpdate(
      suspiciousFaces.length > 0 ? "suspicious" : "authentic",
      totalFaces > 0 ? activeList[0].confidence : 100.0,
      primaryEmotion
    );
  }

  // Throttled Status Broadcast to service worker
  let lastBroadcastStatus = "";
  let lastBroadcastTime = 0;

  function broadcastStatusUpdate(verdict, confidence, emotion) {
    const key = `${verdict}_${Math.round(confidence / 5)}_${emotion}`;
    const now = performance.now();
    if (key === lastBroadcastStatus && now - lastBroadcastTime < 3000) return;
    lastBroadcastStatus = key;
    lastBroadcastTime = now;

    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: "DETECTION_UPDATE",
        payload: { verdict, confidence, emotion }
      }).catch(() => {});
    }
  }

  function renderOffline() {
    if (hudBadge) {
      hudBadge.className = "hv-hud-badge hv-scanning";
      hudBadge.querySelector("#hv-badge-title").textContent = "SHIELD READY";
      hudBadge.querySelector("#hv-badge-score").textContent = "Connecting to backend...";
    }
    for (const [id, face] of trackedFaces.entries()) {
      if (face.overlayElement && face.overlayElement.parentNode) {
        face.overlayElement.parentNode.removeChild(face.overlayElement);
      }
    }
    trackedFaces.clear();
  }

  function hideHUD() {
    if (hudBadge) hudBadge.style.display = "none";
    for (const [id, face] of trackedFaces.entries()) {
      if (face.overlayElement && face.overlayElement.parentNode) {
        face.overlayElement.parentNode.removeChild(face.overlayElement);
      }
    }
    trackedFaces.clear();
  }

  // Draggable Dock Pill
  function createDockPill() {
    if (dockPill || document.getElementById("hv-dock-pill")) return;

    dockPill = document.createElement("div");
    dockPill.id = "hv-dock-pill";
    dockPill.className = "hv-dock-pill";

    dockPill.innerHTML = `
      <div class="hv-drag-handle" title="Drag to move shield across meeting screen">⋮⋮</div>
      <div class="hv-shield-icon">
        <svg viewBox="0 0 24 24">
          <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-1 6h2v6h-2V7zm0 8h2v2h-2v-2z"/>
        </svg>
      </div>
      <div class="hv-status-label">
        <span id="hv-dock-text">Deepfake Shield Active</span>
        <span class="hv-sub" id="hv-dock-sub">Voice, Video &amp; Emotion Shield Active</span>
      </div>
      <button class="hv-btn-toggle hv-active" id="hv-btn-dock-toggle">Shield ON</button>
    `;

    document.body.appendChild(dockPill);

    dockPill.addEventListener("click", (e) => {
      if (e.target && (e.target.id === "hv-btn-dock-toggle" || e.target.closest("#hv-btn-dock-toggle"))) return;
      const v = findPrimaryVideo();
      if (v) {
        activeVideo = v;
        attachHUD(v);
        attachVideoObservers(v);
        executeLiveScan();
      }
    });

    dockPill.querySelector("#hv-btn-dock-toggle").addEventListener("click", (e) => {
      e.stopPropagation();
      config.enabled = !config.enabled;
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ enabled: config.enabled });
      }
      updateDockPill();
      if (!config.enabled) {
        hideHUD();
      } else {
        startDetectionLoop();
      }
    });

    makeDraggable(dockPill);
    updateDockPill();
  }

  function makeDraggable(element) {
    let isDragging = false;
    let startX, startY, initialLeft, initialTop;

    function onPointerDown(e) {
      if (e.target && e.target.tagName === "BUTTON") return;
      isDragging = true;
      element.classList.add("hv-dragging");

      const rect = element.getBoundingClientRect();
      initialLeft = rect.left;
      initialTop = rect.top;
      startX = e.clientX || (e.touches && e.touches[0].clientX);
      startY = e.clientY || (e.touches && e.touches[0].clientY);

      element.style.bottom = "auto";
      element.style.right = "auto";
      element.style.left = `${initialLeft}px`;
      element.style.top = `${initialTop}px`;

      e.preventDefault();
    }

    function onPointerMove(e) {
      if (!isDragging) return;
      const clientX = e.clientX || (e.touches && e.touches[0].clientX);
      const clientY = e.clientY || (e.touches && e.touches[0].clientY);

      const dx = clientX - startX;
      const dy = clientY - startY;

      let newLeft = initialLeft + dx;
      let newTop = initialTop + dy;

      const maxLeft = window.innerWidth - element.offsetWidth - 10;
      const maxTop = window.innerHeight - element.offsetHeight - 10;
      newLeft = Math.max(10, Math.min(newLeft, maxLeft));
      newTop = Math.max(10, Math.min(newTop, maxTop));

      element.style.left = `${newLeft}px`;
      element.style.top = `${newTop}px`;
    }

    function onPointerUp() {
      if (isDragging) {
        isDragging = false;
        element.classList.remove("hv-dragging");
        document.removeEventListener("mousemove", onPointerMove);
        document.removeEventListener("mouseup", onPointerUp);
        document.removeEventListener("touchmove", onPointerMove);
        document.removeEventListener("touchend", onPointerUp);
      }
    }

    element.addEventListener("mousedown", (e) => {
      onPointerDown(e);
      document.addEventListener("mousemove", onPointerMove);
      document.addEventListener("mouseup", onPointerUp);
    });

    element.addEventListener("touchstart", (e) => {
      onPointerDown(e);
      document.addEventListener("touchmove", onPointerMove, { passive: false });
      document.addEventListener("touchend", onPointerUp);
    }, { passive: false });
  }

  function updateDockPill() {
    if (!dockPill) return;
    const btn = dockPill.querySelector("#hv-btn-dock-toggle");
    const label = dockPill.querySelector("#hv-dock-text");
    const sub = dockPill.querySelector("#hv-dock-sub");

    if (config.enabled) {
      btn.className = "hv-btn-toggle hv-active";
      btn.textContent = "Shield ON";
      label.textContent = "Deepfake Shield Active";
      sub.textContent = "Voice, Video & Emotion Shield Active";
    } else {
      btn.className = "hv-btn-toggle";
      btn.textContent = "Paused";
      label.textContent = "Shield Paused";
      sub.textContent = "Click to resume protection";
    }
  }

  window.addEventListener("resize", () => {
    repositionTrackedFaces();
    updateHUDBadgePosition();
  });

  // Global MutationObserver: Detects video element creation/swaps & style/transform changes in Google Meet
  activeVideoMutationObserver = new MutationObserver(() => {
    if (document.hidden || !config.enabled) return;
    if (!activeVideo || !document.body.contains(activeVideo) || activeVideo.videoWidth < 40) {
      const v = findPrimaryVideo();
      if (v && v !== activeVideo) {
        activeVideo = v;
        attachHUD(v);
        attachVideoObservers(v);
        executeLiveScan();
      }
    } else {
      repositionTrackedFaces();
      updateHUDBadgePosition();
    }
  });
  activeVideoMutationObserver.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["style", "class"] });

  // Start the continuous scanning loop
  startDetectionLoop();
  createDockPill();
})();
