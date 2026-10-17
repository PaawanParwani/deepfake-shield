// Human Verification — Real-Time Google Meet & Video Call Shield
(function () {
  "use strict";

  if (window.__HV_SHIELD_INJECTED__) return;
  window.__HV_SHIELD_INJECTED__ = true;

  console.log("%c[Deepfake Shield]%c Zero-Lag Meeting Shield Active with Emotion & Biometric Tracking", "color: #3b82f6; font-weight: bold;", "color: inherit;");

  let config = {
    enabled: true,
    backendUrl: "https://deepfake-shield-vyaq.onrender.com",
    scanIntervalMs: 220,        // Rapid 0.22s polling for instant discovery
    scanIntervalLockedMs: 600,  // Continuous 0.6s verification when locked
    sensitivityThreshold: 0.88,
    debug: true
  };

  // State
  let inFlight = false;
  let activeVideo = null;
  let hudBadge = null;
  let overlayRoot = null;
  let dockPill = null;
  let animFrameId = null;
  let consecutiveNoFaceCount = 0;
  let candidateIndex = 0;

  // Active Multi-Face Tracking Map: id -> TrackedFace
  const trackedFaces = new Map();
  let faceIdCounter = 0;

  // Local 60 FPS Optical Follower Canvas (128x96 for sub-pixel pattern matching)
  const TRACK_W = 128;
  const TRACK_H = 96;
  const trackCanvas = document.createElement("canvas");
  trackCanvas.width = TRACK_W;
  trackCanvas.height = TRACK_H;
  const trackCtx = trackCanvas.getContext("2d", { willReadFrequently: true });
  let lastOpticalTrackTime = 0;

  // Diagnostic Debug Logger
  function debugLog(...args) {
    if (config.debug) {
      console.log("%c[Deepfake Shield Debug]", "color: #0ea5e9; font-weight: bold;", ...args);
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
      ["enabled", "backendUrl", "scanIntervalMs", "sensitivityThreshold", "debug"],
      (stored) => {
        if (stored) config = { ...config, ...stored };
        updateDockPill();
      }
    );

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local") {
        for (let k in changes) config[k] = changes[k].newValue;
        updateDockPill();
      }
    });
  }

  // Ensure fixed Viewport Overlay Root exists (immune to parent overflow / transforms)
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

  // High-Resolution 540p Compact Frame Capture (~25KB, pristine facial texture)
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  function captureCompactFrame(video) {
    if (!video || video.paused || video.ended || video.readyState < 1) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || vw < 40 || vh < 40) return null;

    try {
      const maxDim = 540;
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
        dataUrl: offscreenCanvas.toDataURL("image/jpeg", 0.82),
        capturedW: targetW,
        capturedH: targetH
      };
    } catch (err) {
      debugLog("Frame capture exception:", err);
      return null;
    }
  }

  // Robust Mirroring Detection (Inspects entire ancestor DOM chain up to body)
  function isVideoMirrored(video) {
    let el = video;
    let mirrorCount = 0;
    while (el && el !== document.body) {
      try {
        const style = window.getComputedStyle(el);
        const t = style.transform || style.webkitTransform || "";
        if (t && t !== "none") {
          if (t.includes("scaleX(-1)") || t.includes("scale(-1")) {
            mirrorCount++;
          } else if (t.startsWith("matrix(")) {
            const parts = t.slice(7, -1).split(",").map((s) => parseFloat(s.trim()));
            if (parts.length >= 1 && parts[0] < 0) {
              mirrorCount++;
            }
          } else if (t.startsWith("matrix3d(")) {
            const parts = t.slice(9, -1).split(",").map((s) => parseFloat(s.trim()));
            if (parts.length >= 1 && parts[0] < 0) {
              mirrorCount++;
            }
          }
        }
      } catch {
        // ignore cross-boundary styling errors
      }
      el = el.parentElement;
    }
    return mirrorCount % 2 === 1;
  }

  // Exact Rendered Video Rect with object-fit: cover / contain / fill
  function getRenderedVideoGeometry(video, fw, fh) {
    const rect = video.getBoundingClientRect();
    const elemW = rect.width;
    const elemH = rect.height;

    if (!fw || !fh || fw <= 0 || fh <= 0) {
      return {
        left: rect.left,
        top: rect.top,
        width: elemW,
        height: elemH,
        renderW: elemW,
        renderH: elemH,
        offsetX: 0,
        offsetY: 0,
        objectFit: "cover"
      };
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
    } else if (objectFit === "cover") {
      const scale = Math.max(elemW / fw, elemH / fh);
      renderW = fw * scale;
      renderH = fh * scale;
      offsetX = (elemW - renderW) / 2;
      offsetY = (elemH - renderH) / 2;
    } else if (objectFit === "fill") {
      renderW = elemW;
      renderH = elemH;
      offsetX = 0;
      offsetY = 0;
    } else {
      const scale = Math.max(elemW / fw, elemH / fh);
      renderW = fw * scale;
      renderH = fh * scale;
      offsetX = (elemW - renderW) / 2;
      offsetY = (elemH - renderH) / 2;
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
      objectFit
    };
  }

  // Strictly verify element visibility
  function isElementVisible(el) {
    if (!el || !document.body.contains(el)) return false;
    try {
      const style = window.getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden" || parseFloat(style.opacity || "1") < 0.1) return false;
      const rect = el.getBoundingClientRect();
      if (rect.width < 70 || rect.height < 50) return false;
      if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) return false;
      return true;
    } catch {
      return false;
    }
  }

  // Get all active, visible video tiles on screen
  function getCandidateVideos() {
    const videos = Array.from(document.querySelectorAll("video"));
    return videos.filter((v) => {
      if (v.paused || v.ended || v.readyState < 1) return false;
      if (!v.videoWidth || !v.videoHeight || v.videoWidth < 40 || v.videoHeight < 40) return false;
      return isElementVisible(v);
    });
  }

  // Smart Video Selection: Prioritize the primary main stage video (largest active video)
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

    // Sticky lock: If current active video is still valid and at least 70% of maxArea, maintain lock
    // This stops jitter between similarly-sized grid tiles and NEVER cycles away to a tiny preview tile
    if (activeVideo && candidates.includes(activeVideo)) {
      const curRect = activeVideo.getBoundingClientRect();
      const curArea = curRect.width * curRect.height;
      if (curArea >= maxArea * 0.70) {
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
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 9000);

    try {
      const resp = await fetch(`${config.backendUrl}/predict/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: capture.dataUrl }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (resp.ok) {
        const data = await resp.json();
        const clientLatency = Math.round(performance.now() - reqStart);
        try {
          processDetections(data, video, capture.capturedW, capture.capturedH, clientLatency);
        } catch (procErr) {
          console.error("[Deepfake Shield] UI processing exception:", procErr);
        }
      } else {
        debugLog(`Server responded with HTTP ${resp.status}`);
      }
    } catch (err) {
      clearTimeout(timeoutId);
      debugLog("Inference fetch error:", err);
      renderOffline();
    } finally {
      inFlight = false;
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
          <span class="hv-hud-score" id="hv-badge-score">Initializing...</span>
        </div>
      `;
      document.body.appendChild(hudBadge);
    }

    const candidates = getCandidateVideos();
    candidates.forEach((v) => {
      if (!v.__hv_click_bound) {
        v.__hv_click_bound = true;
        v.style.cursor = "pointer";
        v.title = "Click to focus Deepfake Shield on this video tile";
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
    const fw = data.frame_width || capturedW || 540;
    const fh = data.frame_height || capturedH || 360;
    const mirrored = isVideoMirrored(video);
    const geom = getRenderedVideoGeometry(video, fw, fh);
    debugLog(`Video Element: ${Math.round(geom.width)}x${Math.round(geom.height)} | Stream: ${fw}x${fh} | Mirrored: ${mirrored}`);

    // ZERO FACES DETECTED: Instant Zero-Ghost Purge
    if (!face_detected || verdict === "no_face" || !data.faces || data.faces.length === 0) {
      consecutiveNoFaceCount++;
      hudBadge.className = "hv-hud-badge hv-no-face";
      hudBadge.querySelector("#hv-badge-title").textContent = "NO FACE DETECTED";
      hudBadge.querySelector("#hv-badge-score").textContent = "Position face in camera";
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = "Scanning for participant face...";
      }

      // Purge all stale boxes and tags immediately from DOM
      for (const [id, face] of trackedFaces.entries()) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
      }
      trackedFaces.clear();
      return;
    }

    consecutiveNoFaceCount = 0;

    const incomingFaces = data.faces && data.faces.length > 0 ? data.faces : [{
      face_id: "target_1",
      bounding_box: bounding_box,
      raw_box: bounding_box,
      verdict: verdict,
      p_fake: p_fake,
      raw_p_fake: p_fake,
      sharpness: 100,
      confidence: confidence,
      emotion: emotion || "Focused / Calm 👤"
    }];

    debugLog(`Incoming Detections (${latency_ms || clientLatency}ms): ${incomingFaces.length} Face(s)`);

    const now = performance.now();
    const matchedTrackIds = new Set();

    // Map incoming detections to screen coordinates and frame normalized coords
    const mappedDetections = incomingFaces.map((det, idx) => {
      const box = Array.isArray(det.bounding_box) && det.bounding_box.length === 4
        ? det.bounding_box
        : (Array.isArray(det.raw_box) && det.raw_box.length === 4 ? det.raw_box : [0, 0, fw, fh]);
      const [bx1, by1, bx2, by2] = box;
      const effX1 = mirrored ? (fw - bx2) : bx1;
      const effX2 = mirrored ? (fw - bx1) : bx2;

      const screenX = geom.left + geom.offsetX + (effX1 / fw) * geom.renderW;
      const screenY = geom.top + geom.offsetY + (by1 / fh) * geom.renderH;
      const rawW = ((effX2 - effX1) / fw) * geom.renderW;
      const rawH = ((by2 - by1) / fh) * geom.renderH;
      // Proportional face clamp: height capped at 1.30x width (natural portrait ratio, covers chin and beard)
      const screenW = rawW;
      const screenH = Math.min(rawH, screenW * 1.30);
      const centerX = screenX + screenW / 2;
      const centerY = screenY + screenH / 2;

      // Center in raw video coordinates (0.0 - 1.0) for optical template tracking
      const normCx = ((bx1 + bx2) / 2) / fw;
      const normCy = ((by1 + by2) / 2) / fh;

      return {
        det,
        targetIndex: idx + 1,
        screenX,
        screenY,
        screenW,
        screenH,
        centerX,
        centerY,
        normCx,
        normCy
      };
    });

    // Frontend Non-Maximum Suppression (Eliminates spurious ear/cap duplicate boxes)
    const filteredDetections = [];
    mappedDetections.sort((a, b) => (b.screenW * b.screenH) - (a.screenW * a.screenH));
    for (const det of mappedDetections) {
      let keep = true;
      for (const kept of filteredDetections) {
        const xA = Math.max(det.screenX, kept.screenX);
        const yA = Math.max(det.screenY, kept.screenY);
        const xB = Math.min(det.screenX + det.screenW, kept.screenX + kept.screenW);
        const yB = Math.min(det.screenY + det.screenH, kept.screenY + kept.screenH);
        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const unionArea = (det.screenW * det.screenH) + (kept.screenW * kept.screenH) - interArea;
        const iou = unionArea > 0 ? interArea / unionArea : 0;
        const centerDist = Math.hypot(det.centerX - kept.centerX, det.centerY - kept.centerY);
        if (iou > 0.15 || centerDist < Math.max(det.screenW, kept.screenW) * 0.70) {
          keep = false;
          break;
        }
      }
      if (keep) filteredDetections.push(det);
    }

    // 1-TO-1 MATCHING (Prevents ghost boxes)
    filteredDetections.forEach((mDet) => {
      let bestMatch = null;

      if (filteredDetections.length === 1 && trackedFaces.size >= 1) {
        // Strict single face lock: bind directly to primary face track
        bestMatch = Array.from(trackedFaces.values())[0];
      } else {
        let minDistance = Infinity;
        for (const [id, face] of trackedFaces.entries()) {
          if (matchedTrackIds.has(id)) continue;
          const curCX = face.current.x + face.current.w / 2;
          const curCY = face.current.y + face.current.h / 2;
          const dist = Math.hypot(curCX - mDet.centerX, curCY - mDet.centerY);
          if (dist < minDistance) {
            minDistance = dist;
            bestMatch = face;
          }
        }
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
          history: [],
          smoothedScore: mDet.det.p_fake || 0.0,
          consecutiveHighRisk: 0,
          verdict: mDet.det.verdict || "authentic",
          confidence: mDet.det.confidence || 100.0,
          sharpness: mDet.det.sharpness || 100,
          emotion: mDet.det.emotion || "Focused / Calm 👤",
          lastSeen: now,
          opacity: 1.0,
          overlayElement: overlayEl,
          tagElement: tagEl,
          template: null,
          templateNormPos: { cx: mDet.normCx, cy: mDet.normCy }
        };
        trackedFaces.set(newId, bestMatch);
      }

      matchedTrackIds.add(bestMatch.id);
      bestMatch.lastSeen = now;
      bestMatch.emotion = mDet.det.emotion || "Focused / Calm 👤";
      bestMatch.label = `Face #${mDet.targetIndex}`;
      bestMatch.target = { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH };
      bestMatch.sharpness = mDet.det.sharpness || 100;
      bestMatch.templateNormPos = { cx: mDet.normCx, cy: mDet.normCy };
      bestMatch.opacity = 1.0;

      // Extract fresh optical tracking template on ground-truth update
      extractFaceTemplate(video, bestMatch, mDet.normCx, mDet.normCy);

      // Temporal smoothing (5 frames)
      bestMatch.history.push({
        p_fake: mDet.det.p_fake || 0.0,
        verdict: mDet.det.verdict || "authentic",
        ts: now
      });
      if (bestMatch.history.length > 5) bestMatch.history.shift();

      const avgScore = bestMatch.history.reduce((acc, h) => acc + h.p_fake, 0) / bestMatch.history.length;
      bestMatch.smoothedScore = avgScore;

      if ((mDet.det.p_fake || 0) >= 0.88) {
        bestMatch.consecutiveHighRisk++;
      } else {
        bestMatch.consecutiveHighRisk = Math.max(0, bestMatch.consecutiveHighRisk - 1);
      }

      if (avgScore >= 0.88 && bestMatch.consecutiveHighRisk >= 4) {
        bestMatch.verdict = "suspicious";
        bestMatch.confidence = Math.round(avgScore * 100);
      } else if (avgScore <= 0.40) {
        bestMatch.verdict = "authentic";
        bestMatch.confidence = Math.round((1.0 - avgScore) * 100);
      } else {
        bestMatch.verdict = "uncertain";
        bestMatch.confidence = Math.round(avgScore * 100);
      }
    });

    for (const [id, face] of trackedFaces.entries()) {
      if (!matchedTrackIds.has(id)) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
        trackedFaces.delete(id);
      }
    }

    // Determine Aggregate Verdict for Floating Top HUD Badge
    const activeList = Array.from(trackedFaces.values());
    const totalFaces = activeList.length;
    const suspiciousFaces = activeList.filter((f) => f.verdict === "suspicious");
    const uncertainFaces = activeList.filter((f) => f.verdict === "uncertain");
    const displayName = "PARTICIPANT";

    const primaryEmotion = (activeList.length > 0 && activeList[0].emotion) || emotion || "Focused / Calm 👤";

    if (suspiciousFaces.length > 0) {
      const maxRisk = Math.max(...suspiciousFaces.map((f) => f.confidence));
      hudBadge.className = "hv-hud-badge hv-suspicious";
      hudBadge.querySelector("#hv-badge-title").textContent = totalFaces > 1 ? `DEEPFAKE ALERT (${suspiciousFaces.length}/${totalFaces})` : "DEEPFAKE DETECTED";
      hudBadge.querySelector("#hv-badge-score").textContent = `${maxRisk}% Risk (${latency_ms || clientLatency}ms)`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = `⚠️ Deepfake Alert (${maxRisk}% Risk)`;
      }
    } else if (uncertainFaces.length > 0) {
      hudBadge.className = "hv-hud-badge hv-uncertain";
      hudBadge.querySelector("#hv-badge-title").textContent = totalFaces > 1 ? `ANALYZING (${totalFaces} TARGETS)` : "ANALYZING VIDEO";
      hudBadge.querySelector("#hv-badge-score").textContent = `${primaryEmotion} (${latency_ms || clientLatency}ms)`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = `Analyzing Video Consistency • ${primaryEmotion}`;
      }
    } else {
      const minAuthConf = totalFaces > 0 ? Math.min(...activeList.map((f) => f.confidence)) : 100.0;
      hudBadge.className = "hv-hud-badge hv-authentic";
      hudBadge.querySelector("#hv-badge-title").textContent = totalFaces > 1 ? `${displayName} (${totalFaces} TARGETS)` : `${displayName} • VERIFIED`;
      hudBadge.querySelector("#hv-badge-score").textContent = `${minAuthConf}% Authentic • ${primaryEmotion}`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = `${displayName} Protected • ${primaryEmotion}`;
      }
    }

    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: "DETECTION_UPDATE",
        payload: {
          verdict: suspiciousFaces.length > 0 ? "suspicious" : (uncertainFaces.length > 0 ? "uncertain" : "authentic"),
          confidence: totalFaces > 0 ? activeList[0].confidence : 100.0,
          emotion: primaryEmotion
        }
      }).catch(() => {});
    }
  }

  // Extract a 16x16 grayscale template of the face center on the 128x96 tracking canvas
  function extractFaceTemplate(video, face, normCx, normCy) {
    try {
      trackCtx.drawImage(video, 0, 0, TRACK_W, TRACK_H);
      const px = Math.max(8, Math.min(TRACK_W - 9, Math.round(normCx * TRACK_W)));
      const py = Math.max(8, Math.min(TRACK_H - 9, Math.round(normCy * TRACK_H)));

      const patch = trackCtx.getImageData(px - 8, py - 8, 16, 16).data;
      const gray = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        const r = patch[i * 4];
        const g = patch[i * 4 + 1];
        const b = patch[i * 4 + 2];
        gray[i] = (r * 77 + g * 150 + b * 29) >> 8;
      }
      face.template = gray;
      face.templateNormPos = { cx: normCx, cy: normCy };
    } catch {}
  }

  // 60 FPS Optical Template Matching (Tracks head motion between network responses with 0ms lag)
  function runOpticalTemplateTracker(video) {
    if (!video || video.paused || video.ended || video.readyState < 2 || trackedFaces.size === 0 || consecutiveNoFaceCount > 0) return;
    const now = performance.now();
    if (now - lastOpticalTrackTime < 16) return; // ~60 FPS
    lastOpticalTrackTime = now;

    try {
      trackCtx.drawImage(video, 0, 0, TRACK_W, TRACK_H);
      const frameData = trackCtx.getImageData(0, 0, TRACK_W, TRACK_H).data;
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      const geom = getRenderedVideoGeometry(video, vw, vh);
      const mirrored = isVideoMirrored(video);

      for (const face of trackedFaces.values()) {
        if (!face.template) continue;

        const curPX = Math.round(face.templateNormPos.cx * TRACK_W);
        const curPY = Math.round(face.templateNormPos.cy * TRACK_H);

        // Search +/- 10px horizontally, +/- 8px vertically (corresponds to ~80px on 1080p screen)
        let bestSAD = Infinity;
        let bestX = curPX;
        let bestY = curPY;

        for (let dy = -8; dy <= 8; dy += 2) {
          const sy = curPY + dy - 8;
          if (sy < 0 || sy + 16 > TRACK_H) continue;
          for (let dx = -10; dx <= 10; dx += 2) {
            const sx = curPX + dx - 8;
            if (sx < 0 || sx + 16 > TRACK_W) continue;

            let sad = 0;
            for (let ty = 0; ty < 16; ty += 2) {
              const fRow = (sy + ty) * TRACK_W + sx;
              const tRow = ty * 16;
              for (let tx = 0; tx < 16; tx += 2) {
                const pIdx = (fRow + tx) * 4;
                const lum = (frameData[pIdx] * 77 + frameData[pIdx + 1] * 150 + frameData[pIdx + 2] * 29) >> 8;
                sad += Math.abs(lum - face.template[tRow + tx]);
              }
            }

            if (sad < bestSAD) {
              bestSAD = sad;
              bestX = curPX + dx;
              bestY = curPY + dy;
            }
          }
        }

        // 64 sampled pixels: threshold < 35 avg diff per pixel indicates valid face track
        if (bestSAD < 64 * 35) {
          const newNormCx = bestX / TRACK_W;
          const newNormCy = bestY / TRACK_H;
          face.templateNormPos.cx = newNormCx;
          face.templateNormPos.cy = newNormCy;

          // Map normalized video coordinate to current screen pixels
          const effNormX = mirrored ? (1.0 - newNormCx) : newNormCx;
          const screenCX = geom.left + geom.offsetX + effNormX * geom.renderW;
          const screenCY = geom.top + geom.offsetY + newNormCy * geom.renderH;

          face.target.x = screenCX - face.target.w / 2;
          face.target.y = screenCY - face.target.h / 2;
        }
      }
    } catch {}
  }

  // 60 FPS Instant Snap Animation Loop
  function animationLoop() {
    if (config.enabled && activeVideo && document.body.contains(activeVideo)) {
      updateHUDBadgePosition();
      runOpticalTemplateTracker(activeVideo);

      for (const [id, face] of trackedFaces.entries()) {
        const dx = face.target.x - face.current.x;
        const dy = face.target.y - face.current.y;
        const dw = face.target.w - face.current.w;
        const dh = face.target.h - face.current.h;

        // Snappy 0.85 lerp factor on motion for instant 1-frame snap
        const lerpFactor = 0.85;

        face.current.x += dx * lerpFactor;
        face.current.y += dy * lerpFactor;
        face.current.w += dw * lerpFactor;
        face.current.h += dh * lerpFactor;

        face.opacity = Math.min(1.0, face.opacity + 0.25);

        const el = face.overlayElement;
        if (el) {
          el.style.display = "block";
          el.style.left = `${Math.round(face.current.x)}px`;
          el.style.top = `${Math.round(face.current.y)}px`;
          el.style.width = `${Math.round(face.current.w)}px`;
          el.style.height = `${Math.round(face.current.h)}px`;
          el.style.opacity = face.opacity.toFixed(2);

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
          } else if (face.verdict === "uncertain") {
            el.className = "hv-face-overlay hv-uncertain-box";
            if (face.tagElement) {
              face.tagElement.innerHTML = `
                <span class="hv-tag-identity">${escapeHTML(nameDisplay)}</span>
                <span class="hv-tag-verdict hv-v-unc">Analyzing (${face.confidence}%)</span>
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
    } else {
      hideHUD();
    }

    animFrameId = requestAnimationFrame(animationLoop);
  }

  function renderOffline() {
    if (!hudBadge) return;
    hudBadge.className = "hv-hud-badge hv-scanning";
    hudBadge.querySelector("#hv-badge-title").textContent = "SHIELD READY";
    hudBadge.querySelector("#hv-badge-score").textContent = "Connecting to backend...";
    for (const face of trackedFaces.values()) {
      if (face.overlayElement) face.overlayElement.style.display = "none";
    }
  }

  function hideHUD() {
    if (hudBadge) hudBadge.style.display = "none";
    for (const face of trackedFaces.values()) {
      if (face.overlayElement) face.overlayElement.style.display = "none";
    }
  }

  // Draggable Dock Pill (Moveable anywhere on Google Meet screen)
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
        <span class="hv-sub" id="hv-dock-sub">Emotion & Biometric AR Shield Active</span>
      </div>
      <button class="hv-btn-toggle hv-active" id="hv-btn-dock-toggle">Shield ON</button>
    `;

    document.body.appendChild(dockPill);

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
      sub.textContent = "Emotion & Biometric AR Shield Active";
    } else {
      btn.className = "hv-btn-toggle";
      btn.textContent = "Paused";
      label.textContent = "Shield Paused";
      sub.textContent = "Click to resume protection";
    }
  }

  // Fast DOM Scanner (Instant discovery when camera connects)
  const observer = new MutationObserver(() => {
    if (document.hidden) return;
    const v = findPrimaryVideo();
    if (v && v !== activeVideo) {
      activeVideo = v;
      attachHUD(v);
      analyzeLiveFrame();
    }
  });

  observer.observe(document.body || document.documentElement, {
    childList: true,
    subtree: true
  });

  // Start 60 FPS animation loop
  animFrameId = requestAnimationFrame(animationLoop);

  // Dynamic Detection Loop (220ms searching for instant lock, 600ms locked)
  function scheduleNextScan() {
    const isLocked = trackedFaces.size > 0;
    const interval = isLocked ? config.scanIntervalLockedMs : config.scanIntervalMs;
    setTimeout(() => {
      analyzeLiveFrame().finally(scheduleNextScan);
    }, interval);
  }
  scheduleNextScan();

  createDockPill();
})();
