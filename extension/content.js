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
    scanIntervalMs: 1000,       // 1.0s search interval
    scanIntervalLockedMs: 800,  // 0.8s verification interval when locked
    sensitivityThreshold: 0.88,
    mirrorMode: "normal",       // "normal" (auto-detects mirrored video), "flipped", "unmirrored"
    debug: false
  };

  // State
  let inFlight = false;
  let activeVideo = null;
  let hudBadge = null;
  let overlayRoot = null;
  let dockPill = null;
  let consecutiveNoFaceCount = 0;
  let consecutiveBackendFailures = 0;
  let loopRunning = false;

  // Active Multi-Face Tracking Map: id -> TrackedFace
  const trackedFaces = new Map();
  let faceIdCounter = 0;

  function debugLog(...args) {
    if (config.debug) {
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

  // Sync extension storage config
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(
      ["enabled", "backendUrl", "scanIntervalMs", "sensitivityThreshold", "mirrorMode"],
      (stored) => {
        if (stored) {
          if (!stored.backendUrl || stored.backendUrl.includes("localhost") || stored.backendUrl.includes("onrender.com") || stored.backendUrl.includes("deepfake-shield-backend")) {
            stored.backendUrl = "http://127.0.0.1:8000";
            chrome.storage.local.set({ backendUrl: "http://127.0.0.1:8000" });
          }
          if (!stored.mirrorMode) {
            stored.mirrorMode = "normal";
          }
          config = { ...config, ...stored };
        }
        updateDockPill();
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

      return {
        dataUrl: offscreenCanvas.toDataURL("image/jpeg", 0.75),
        capturedW: targetW,
        capturedH: targetH
      };
    } catch (err) {
      return null;
    }
  }

  // High-Precision Mirroring Detection for Google Meet, Zoom, Teams, Webex
  function isVideoMirrored(video) {
    if (!video) return false;

    // Explicit user override if configured
    if (config.mirrorMode === "flipped") return true;
    if (config.mirrorMode === "unmirrored") return false;

    let el = video;
    let mirrorCount = 0;
    while (el && el !== document.documentElement) {
      try {
        // 1. Direct inline style check
        const inlineT = (el.style && el.style.transform) ? el.style.transform.toLowerCase() : "";
        if (
          inlineT.includes("scalex(-1)") ||
          inlineT.includes("scale(-1") ||
          inlineT.includes("rotatey(180deg)") ||
          inlineT.includes("matrix(-1")
        ) {
          mirrorCount++;
        } else {
          // 2. Computed style transform check (in Chromium, scaleX(-1) computes to matrix(-1, 0, 0, 1, 0, 0))
          const compStyle = window.getComputedStyle(el);
          const t = (compStyle.transform || compStyle.webkitTransform || "").toLowerCase();
          if (t && t !== "none") {
            if (
              t.includes("scalex(-1)") ||
              t.includes("scale(-1") ||
              t.includes("rotatey(180deg)")
            ) {
              mirrorCount++;
            } else if (t.startsWith("matrix(")) {
              const match = t.match(/^matrix\(\s*([-\d.]+)/);
              if (match && parseFloat(match[1]) < -0.1) {
                mirrorCount++;
              }
            } else if (t.startsWith("matrix3d(")) {
              const match = t.match(/^matrix3d\(\s*([-\d.]+)/);
              if (match && parseFloat(match[1]) < -0.1) {
                mirrorCount++;
              }
            }
          }
        }
      } catch {}
      el = el.parentElement;
    }

    // 3. Google Meet self-view detection:
    // On meet.google.com, local self-view feeds are always mirrored.
    if (mirrorCount % 2 === 0 && window.location.hostname.includes("meet.google.com")) {
      const isSelf = !!(
        video.closest("[data-self-name]") ||
        video.closest("[data-is-self-view='true']") ||
        video.getAttribute("data-is-self-view") === "true" ||
        video.className.includes("self") ||
        (video.parentElement && video.parentElement.className.includes("self"))
      );
      if (isSelf) return true;

      // Single-video solo room check on Google Meet
      const allMeetVideos = document.querySelectorAll("video");
      if (allMeetVideos.length === 1) return true;
    }

    return mirrorCount % 2 === 1;
  }

  // Exact Rendered Video Geometry with object-fit compensation
  function getRenderedVideoGeometry(video, fw, fh) {
    const rect = video.getBoundingClientRect();
    const elemW = rect.width;
    const elemH = rect.height;

    if (!fw || !fh || fw <= 0 || fh <= 0) {
      return { left: rect.left, top: rect.top, width: elemW, height: elemH, renderW: elemW, renderH: elemH, offsetX: 0, offsetY: 0 };
    }

    const style = window.getComputedStyle(video);
    const objectFit = style.objectFit || "cover";

    let renderW = elemW;
    let renderH = elemH;
    let offsetX = 0;
    let offsetY = 0;

    if (objectFit === "contain") {
      const scale = Math.min(elemW / fw, elemH / fh);
      renderW = fw * scale;
      renderH = fh * scale;
      offsetX = (elemW - renderW) / 2;
      offsetY = (elemH - renderH) / 2;
    } else {
      const scale = Math.max(elemW / fw, elemH / fh);
      renderW = fw * scale;
      renderH = fh * scale;
      offsetX = (elemW - renderW) / 2;
      offsetY = (elemH - renderH) / 2;
    }

    return { left: rect.left, top: rect.top, width: elemW, height: elemH, renderW, renderH, offsetX, offsetY };
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

  function findPrimaryVideo() {
    const candidates = getCandidateVideos();
    if (candidates.length === 0) return null;

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
      }

      const capture = captureCompactFrame(video);
      if (!capture) {
        resolve();
        return;
      }

      inFlight = true;
      const reqStart = performance.now();

      // Generous watchdog timeout (7500ms) to tolerate background CPU usage
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

      if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
        try {
          chrome.runtime.sendMessage(
            {
              type: "PREDICT_FRAME",
              payload: { backendUrl: config.backendUrl, image_base64: capture.dataUrl }
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
                processDetections(response.data, video, capture.capturedW, capture.capturedH, clientLatency);
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
        <button id="hv-btn-flip" title="Toggle Mirror / Flip (⇄)" class="hv-badge-flip-btn">⇄ Flip</button>
      `;
      document.body.appendChild(hudBadge);

      const flipBtn = hudBadge.querySelector("#hv-btn-flip");
      if (flipBtn) {
        flipBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const curMirrored = activeVideo ? isVideoMirrored(activeVideo) : false;
          const currentFlip = config.mirrorMode === "flipped" ? true :
                              config.mirrorMode === "unmirrored" ? false :
                              curMirrored;

          // Toggle to opposite
          config.mirrorMode = currentFlip ? "unmirrored" : "flipped";
          if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({ mirrorMode: config.mirrorMode });
          }
          flipBtn.textContent = config.mirrorMode === "flipped" ? "⇄ Flipped" : "⇄ Normal";
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

  // Multi-Face Tracking & Coordinate Mapping
  function processDetections(data, video, capturedW, capturedH, clientLatency) {
    if (!hudBadge) return;
    const root = getOrCreateOverlayRoot();

    const { verdict, face_detected, latency_ms, emotion } = data;
    const fw = data.frame_width || capturedW || 320;
    const fh = data.frame_height || capturedH || 240;

    // High-precision mirroring alignment:
    // When mirrorMode is "normal", automatically adapt to whether the video is mirrored on-screen!
    const isMirrored = isVideoMirrored(video);
    const shouldFlip = config.mirrorMode === "flipped" ? true :
                       config.mirrorMode === "unmirrored" ? false :
                       isMirrored;

    const geom = getRenderedVideoGeometry(video, fw, fh);

    // ZERO FACES DETECTED: Instant Ghost Box Purge
    if (!face_detected || verdict === "no_face" || !data.faces || data.faces.length === 0) {
      consecutiveNoFaceCount++;
      hudBadge.className = "hv-hud-badge hv-no-face";
      hudBadge.querySelector("#hv-badge-title").textContent = "NO FACE DETECTED";
      hudBadge.querySelector("#hv-badge-score").textContent = "Position face in camera";
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = "Scanning for participant face...";
      }

      // Purge all stale boxes immediately
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
      const bx1 = box[0];
      const by1 = box[1];
      const bx2 = box[2];
      const by2 = box[3];

      // Mirror flip X coordinates when video is mirrored (e.g. Google Meet self-view)
      const effX1 = shouldFlip ? (fw - bx2) : bx1;
      const effX2 = shouldFlip ? (fw - bx1) : bx2;

      const screenX = geom.left + geom.offsetX + (effX1 / fw) * geom.renderW;
      const screenY = geom.top + geom.offsetY + (by1 / fh) * geom.renderH;
      const screenW = ((effX2 - effX1) / fw) * geom.renderW;
      const rawH = ((by2 - by1) / fh) * geom.renderH;
      const screenH = Math.min(rawH, screenW * 1.30);
      const centerX = screenX + screenW / 2;
      const centerY = screenY + screenH / 2;

      return { det, targetIndex: idx + 1, screenX, screenY, screenW, screenH, centerX, centerY };
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

        root.appendChild(overlayEl);

        bestMatch = {
          id: newId,
          label: `Face #${mDet.targetIndex}`,
          overlayElement: overlayEl,
          tagElement: tagEl
        };

        trackedFaces.set(newId, bestMatch);
      }

      matchedTrackIds.add(bestMatch.id);
      bestMatch.emotion = mDet.det.emotion || "Focused / Calm 👤";
      bestMatch.verdict = mDet.det.verdict || "authentic";
      bestMatch.confidence = mDet.det.confidence || 100.0;

      // Direct GPU-accelerated CSS positioning (Zero main-thread layout thrashing)
      const el = bestMatch.overlayElement;
      if (el) {
        el.style.display = "block";
        el.style.left = `${Math.round(mDet.screenX)}px`;
        el.style.top = `${Math.round(mDet.screenY)}px`;
        el.style.width = `${Math.round(mDet.screenW)}px`;
        el.style.height = `${Math.round(mDet.screenH)}px`;
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

    // Keep top HUD position synced
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

    // Throttled status broadcast to background
    broadcastStatusUpdate(
      suspiciousFaces.length > 0 ? "suspicious" : "authentic",
      totalFaces > 0 ? activeList[0].confidence : 100.0,
      primaryEmotion
    );
  }

  // Throttled Status Broadcast
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
    // Clean up all ghost boxes immediately
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

  window.addEventListener("resize", updateHUDBadgePosition);

  // Throttled check for camera reconnects if video disconnects
  setInterval(() => {
    if (document.hidden || !config.enabled) return;
    if (!activeVideo || !document.body.contains(activeVideo)) {
      const v = findPrimaryVideo();
      if (v) {
        activeVideo = v;
        attachHUD(v);
        executeLiveScan();
      }
    }
  }, 2500);

  // Start the rock-solid continuous scanning loop
  startDetectionLoop();
  createDockPill();
})();
