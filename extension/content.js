// Human Verification — Real-Time Google Meet & Video Call Shield
(function () {
  "use strict";

  // 1. NEVER inject into sub-frames or iframes
  if (window !== window.top) return;
  if (window.__HV_SHIELD_INJECTED__) return;
  window.__HV_SHIELD_INJECTED__ = true;

  let config = {
    enabled: true,
    backendUrl: "http://localhost:8000",
    scanIntervalMs: 1000,       // 1.0s search interval
    scanIntervalLockedMs: 750,  // 0.75s verification interval when locked
    sensitivityThreshold: 0.88,
    mirrorMode: "auto",         // "auto", "mirrored", "unmirrored"
    debug: false
  };

  // State
  let inFlight = false;
  let activeVideo = null;
  let hudBadge = null;
  let overlayRoot = null;
  let dockPill = null;
  let animFrameId = null;
  let scanTimer = null;
  let consecutiveNoFaceCount = 0;

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

  // Sync extension storage config
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(
      ["enabled", "backendUrl", "scanIntervalMs", "sensitivityThreshold", "mirrorMode"],
      (stored) => {
        if (stored) {
          if (stored.backendUrl && (stored.backendUrl.includes("onrender.com") || stored.backendUrl.includes("deepfake-shield-backend"))) {
            stored.backendUrl = "http://localhost:8000";
            chrome.storage.local.set({ backendUrl: "http://localhost:8000" });
          }
          config = { ...config, ...stored };
        }
        updateDockPill();
      }
    );

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local") {
        for (let k in changes) {
          if (k === "backendUrl" && changes[k].newValue && (changes[k].newValue.includes("onrender.com") || changes[k].newValue.includes("deepfake-shield-backend"))) {
            config.backendUrl = "http://localhost:8000";
          } else {
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

  // Lightweight 360p Compact Frame Capture (~12KB, zero system lag)
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  function captureCompactFrame(video) {
    if (!video || video.paused || video.ended || video.readyState < 2) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || vw < 40 || vh < 40) return null;

    try {
      const maxDim = 360;
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
    if (config.mirrorMode === "mirrored") return true;
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
              // 2D affine transform matrix(a, b, c, d, tx, ty) where 'a' is horizontal scale
              const match = t.match(/^matrix\(\s*([-\d.]+)/);
              if (match && parseFloat(match[1]) < -0.1) {
                mirrorCount++;
              }
            } else if (t.startsWith("matrix3d(")) {
              // 3D transform matrix3d(m11, ...) where 'm11' is horizontal scale
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

  // Exact Rendered Video Geometry
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

  // Main Live Frame Analysis
  async function analyzeLiveFrame() {
    if (document.hidden || !config.enabled || inFlight) return;

    const video = findPrimaryVideo();
    if (!video) {
      hideHUD();
      return;
    }

    if (activeVideo !== video) {
      activeVideo = video;
      attachHUD(video);
    }

    const capture = captureCompactFrame(video);
    if (!capture) return;

    inFlight = true;
    const reqStart = performance.now();

    // Safety watchdog: 3000ms max
    const watchdogId = setTimeout(() => {
      if (inFlight) {
        inFlight = false;
        renderOffline();
      }
    }, 3000);

    function handleSuccess(data) {
      clearTimeout(watchdogId);
      inFlight = false;
      const clientLatency = Math.round(performance.now() - reqStart);
      try {
        processDetections(data, video, capture.capturedW, capture.capturedH, clientLatency);
      } catch (err) {
        debugLog("Detection process error:", err);
      }
    }

    // Always route through background service worker (NEVER fetch directly in Google Meet origin)
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
              renderOffline();
              scheduleNextScan(3500); // Back off 3.5s when backend is offline
              return;
            }
            handleSuccess(response.data);
          }
        );
        return;
      } catch (sendErr) {
        clearTimeout(watchdogId);
        inFlight = false;
        renderOffline();
        scheduleNextScan(3500);
        return;
      }
    }

    inFlight = false;
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
          const current = isVideoMirrored(activeVideo);
          config.mirrorMode = current ? "unmirrored" : "mirrored";
          if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({ mirrorMode: config.mirrorMode });
          }
          analyzeLiveFrame();
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
          analyzeLiveFrame();
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

    const { p_fake, verdict, confidence, face_detected, bounding_box, latency_ms, emotion } = data;
    const fw = data.frame_width || capturedW || 360;
    const fh = data.frame_height || capturedH || 240;
    const mirrored = isVideoMirrored(video);
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
      return;
    }

    consecutiveNoFaceCount = 0;

    const detections = data.faces;
    const now = performance.now();
    const matchedTrackIds = new Set();

    const mappedDetections = detections.map((det, idx) => {
      const box = det.bounding_box || [0, 0, 0, 0];
      const bx1 = box[0];
      const by1 = box[1];
      const bx2 = box[2];
      const by2 = box[3];

      const effX1 = mirrored ? (fw - bx2) : bx1;
      const effX2 = mirrored ? (fw - bx1) : bx2;

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
          current: { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH },
          target: { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH },
          smoothedScore: mDet.det.p_fake || 0.0,
          verdict: mDet.det.verdict || "authentic",
          confidence: mDet.det.confidence || 100.0,
          emotion: mDet.det.emotion || "Focused / Calm 👤",
          overlayElement: overlayEl,
          tagElement: tagEl
        };

        trackedFaces.set(newId, bestMatch);
      }

      matchedTrackIds.add(bestMatch.id);
      bestMatch.emotion = mDet.det.emotion || "Focused / Calm 👤";
      bestMatch.target = { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH };
      bestMatch.verdict = mDet.det.verdict || "authentic";
      bestMatch.confidence = mDet.det.confidence || 100.0;
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

    // Start lerp animation loop only when faces are present
    ensureAnimationRunning();

    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: "DETECTION_UPDATE",
        payload: {
          verdict: suspiciousFaces.length > 0 ? "suspicious" : "authentic",
          confidence: totalFaces > 0 ? activeList[0].confidence : 100.0,
          emotion: primaryEmotion
        }
      }).catch(() => {});
    }
  }

  // Smooth lerp animation loop (Only runs when faces are active on screen)
  function animationLoop() {
    if (config.enabled && activeVideo && document.body.contains(activeVideo)) {
      updateHUDBadgePosition();

      if (trackedFaces.size > 0) {
        for (const [id, face] of trackedFaces.entries()) {
          const dx = face.target.x - face.current.x;
          const dy = face.target.y - face.current.y;
          const dw = face.target.w - face.current.w;
          const dh = face.target.h - face.current.h;

          // If box jumped across screen (e.g. mirror flip or initial detection), snap immediately
          if (Math.abs(dx) > 150) {
            face.current.x = face.target.x;
            face.current.y = face.target.y;
            face.current.w = face.target.w;
            face.current.h = face.target.h;
          } else {
            // Smooth lerp (0.42)
            face.current.x += dx * 0.42;
            face.current.y += dy * 0.42;
            face.current.w += dw * 0.42;
            face.current.h += dh * 0.42;
          }

          const el = face.overlayElement;
          if (el) {
            el.style.display = "block";
            el.style.left = `${Math.round(face.current.x)}px`;
            el.style.top = `${Math.round(face.current.y)}px`;
            el.style.width = `${Math.round(face.current.w)}px`;
            el.style.height = `${Math.round(face.current.h)}px`;
            el.style.opacity = "1";

            const nameDisplay = face.label || "Face";
            const emotionDisplay = face.emotion || "Focused / Calm 👤";

            if (face.verdict === "suspicious") {
              el.className = "hv-face-overlay hv-suspicious-box";
              if (face.tagElement) {
                face.tagElement.innerHTML = `
                  <span class="hv-tag-identity">${escapeHTML(nameDisplay)}</span>
                  <span class="hv-tag-verdict hv-v-susp">${face.confidence}% Risk (Deepfake)</span>
                  <span class="hv-tag-emotion">${escapeHTML(emotionDisplay)}</span>
                `;
              }
            } else {
              el.className = "hv-face-overlay";
              if (face.tagElement) {
                face.tagElement.innerHTML = `
                  <span class="hv-tag-identity">${escapeHTML(nameDisplay)}</span>
                  <span class="hv-tag-verdict hv-v-auth">${face.confidence}% Real</span>
                  <span class="hv-tag-emotion">${escapeHTML(emotionDisplay)}</span>
                `;
              }
            }
          }
        }
        animFrameId = requestAnimationFrame(animationLoop);
        return;
      }
    } else {
      hideHUD();
    }
    animFrameId = null;
  }

  function ensureAnimationRunning() {
    if (!animFrameId && trackedFaces.size > 0) {
      animFrameId = requestAnimationFrame(animationLoop);
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
    animFrameId = null;
  }

  function hideHUD() {
    if (hudBadge) hudBadge.style.display = "none";
    for (const [id, face] of trackedFaces.entries()) {
      if (face.overlayElement && face.overlayElement.parentNode) {
        face.overlayElement.parentNode.removeChild(face.overlayElement);
      }
    }
    trackedFaces.clear();
    animFrameId = null;
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
        analyzeLiveFrame();
      }
    });

    dockPill.querySelector("#hv-btn-dock-toggle").addEventListener("click", (e) => {
      e.stopPropagation();
      config.enabled = !config.enabled;
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ enabled: config.enabled });
      }
      updateDockPill();
      if (!config.enabled) hideHUD();
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
      }
    }

    element.addEventListener("mousedown", onPointerDown);
    document.addEventListener("mousemove", onPointerMove);
    document.addEventListener("mouseup", onPointerUp);

    element.addEventListener("touchstart", onPointerDown, { passive: false });
    document.addEventListener("touchmove", onPointerMove, { passive: false });
    document.addEventListener("touchend", onPointerUp);
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

  // Throttled 2-second check for camera connects (no heavy DOM observer)
  setInterval(() => {
    if (document.hidden || !config.enabled) return;
    const v = findPrimaryVideo();
    if (v && v !== activeVideo) {
      activeVideo = v;
      attachHUD(v);
      analyzeLiveFrame();
    }
  }, 2000);

  // Dynamic Detection Loop (1.0s search, 0.75s locked, 3.5s offline backoff)
  function scheduleNextScan(delayMs) {
    if (scanTimer) clearTimeout(scanTimer);
    const isLocked = trackedFaces.size > 0;
    const interval = delayMs || (isLocked ? config.scanIntervalLockedMs : config.scanIntervalMs);
    scanTimer = setTimeout(async () => {
      try {
        await analyzeLiveFrame();
      } finally {
        scheduleNextScan();
      }
    }, interval);
  }
  scheduleNextScan();

  createDockPill();
})();
