// Human Verification — Real-Time Google Meet & Video Call Shield
(function () {
  "use strict";

  if (window.__HV_SHIELD_INJECTED__) return;
  window.__HV_SHIELD_INJECTED__ = true;

  console.log("%c[Deepfake Shield]%c Real-Time Zero-Lag Meeting Shield Active", "color: #3b82f6; font-weight: bold;", "color: inherit;");

  let config = {
    enabled: true,
    backendUrl: "https://deepfake-shield-vyaq.onrender.com",
    scanIntervalMs: 700,        // Fast polling for instant detection
    scanIntervalLockedMs: 1100, // Background verification when locked
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
  let isNativeDetecting = false;
  let consecutiveNoFaceCount = 0;
  let candidateIndex = 0;

  // Active Multi-Face Tracking Map: id -> TrackedFace
  const trackedFaces = new Map();
  let faceIdCounter = 0;

  // Native Chromium FaceDetector (Hardware-Accelerated)
  let nativeFaceDetector = null;
  if (typeof window.FaceDetector !== "undefined") {
    try {
      nativeFaceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 4 });
      console.log("%c[Deepfake Shield]%c Native Chromium FaceDetector active (0ms hardware tracking)", "color: #10b981; font-weight: bold;", "color: inherit;");
    } catch {
      nativeFaceDetector = null;
    }
  }

  // Diagnostic Debug Logger
  function debugLog(...args) {
    if (config.debug) {
      console.log("%c[Deepfake Shield Debug]", "color: #0ea5e9; font-weight: bold;", ...args);
    }
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

  // Pre-allocated offscreen canvas (max 320px for ~9KB ultra-fast network transfer)
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  // High-speed compressed frame capture (< 10KB payload)
  function captureCompactFrame(video) {
    if (!video || video.paused || video.ended || video.readyState < 1) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || vw < 40 || vh < 40) return null;

    try {
      const maxDim = 320;
      const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
      const targetW = Math.max(140, Math.round(vw * scale));
      const targetH = Math.max(100, Math.round(vh * scale));

      if (offscreenCanvas.width !== targetW || offscreenCanvas.height !== targetH) {
        offscreenCanvas.width = targetW;
        offscreenCanvas.height = targetH;
      }

      offscreenCtx.drawImage(video, 0, 0, targetW, targetH);
      return {
        dataUrl: offscreenCanvas.toDataURL("image/jpeg", 0.62),
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

  // Strictly verify element visibility (eliminates hidden background videos in Google Meet)
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

  // Smart Video Selection: Automatically finds the active webcam or participant tile
  function findPrimaryVideo() {
    const candidates = getCandidateVideos();
    if (candidates.length === 0) return null;

    if (activeVideo && candidates.includes(activeVideo)) {
      if (consecutiveNoFaceCount >= 3 && candidates.length > 1) {
        candidateIndex = (candidateIndex + 1) % candidates.length;
        const nextVideo = candidates[candidateIndex];
        debugLog(`Cycling search to next video tile (${candidateIndex + 1}/${candidates.length})`);
        return nextVideo;
      }
      return activeVideo;
    }

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
    try {
      const resp = await fetch(`${config.backendUrl}/predict/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: capture.dataUrl })
      });

      if (resp.ok) {
        const data = await resp.json();
        const clientLatency = Math.round(performance.now() - reqStart);
        processDetections(data, video, capture.capturedW, capture.capturedH, clientLatency);
      } else {
        debugLog(`Server responded with HTTP ${resp.status}`);
      }
    } catch (err) {
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

  // Multi-Face Tracking & Coordinate Mapping (Ghost-Box-Free Centroid Matching)
  function processDetections(data, video, capturedW, capturedH, clientLatency) {
    if (!hudBadge) return;
    const root = getOrCreateOverlayRoot();

    const { p_fake, verdict, confidence, face_detected, bounding_box, latency_ms } = data;
    const fw = data.frame_width || capturedW || 320;
    const fh = data.frame_height || capturedH || 180;
    const mirrored = isVideoMirrored(video);
    const geom = getRenderedVideoGeometry(video, fw, fh);

    debugLog(`Video Element: ${Math.round(geom.width)}x${Math.round(geom.height)} | Stream: ${fw}x${fh} | Fit: ${geom.objectFit} | Mirrored: ${mirrored}`);

    // CASE 1: ZERO FACES DETECTED
    if (!face_detected || verdict === "no_face" || !data.faces || data.faces.length === 0) {
      consecutiveNoFaceCount++;
      hudBadge.className = "hv-hud-badge hv-no-face";
      hudBadge.querySelector("#hv-badge-title").textContent = "NO FACE DETECTED";
      hudBadge.querySelector("#hv-badge-score").textContent = "Position face in camera";
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = "Scanning for participant face...";
      }

      // Promptly clean up all tracked faces
      for (const [id, face] of trackedFaces.entries()) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
        trackedFaces.delete(id);
      }
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
      confidence: confidence
    }];

    debugLog(`Incoming Detections (${latency_ms || clientLatency}ms): ${incomingFaces.length} Face(s)`);

    const now = performance.now();
    const matchedTrackIds = new Set();

    // Map incoming detections to screen coordinates
    const mappedDetections = incomingFaces.map((det, idx) => {
      const [bx1, by1, bx2, by2] = det.bounding_box;
      const effX1 = mirrored ? (fw - bx2) : bx1;
      const effX2 = mirrored ? (fw - bx1) : bx2;

      const screenX = geom.left + geom.offsetX + (effX1 / fw) * geom.renderW;
      const screenY = geom.top + geom.offsetY + (by1 / fh) * geom.renderH;
      const screenW = ((effX2 - effX1) / fw) * geom.renderW;
      const screenH = ((by2 - by1) / fh) * geom.renderH;
      const centerX = screenX + screenW / 2;
      const centerY = screenY + screenH / 2;

      return {
        det,
        targetIndex: idx + 1,
        screenX,
        screenY,
        screenW,
        screenH,
        centerX,
        centerY
      };
    });

    // ROBUST 1-TO-1 MATCHING (Prevents ghost boxes when user moves their head)
    mappedDetections.forEach((mDet) => {
      let bestMatch = null;

      // Special rule: if single face detected and single face tracked, match unconditionally!
      if (mappedDetections.length === 1 && trackedFaces.size === 1) {
        bestMatch = Array.from(trackedFaces.values())[0];
      } else {
        // Multi-face: match by nearest centroid
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
        root.appendChild(overlayEl);

        bestMatch = {
          id: newId,
          label: `Target #${mDet.targetIndex}`,
          current: { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH },
          target: { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH },
          history: [],
          smoothedScore: mDet.det.p_fake || 0.0,
          consecutiveHighRisk: 0,
          verdict: mDet.det.verdict || "authentic",
          confidence: mDet.det.confidence || 90.0,
          sharpness: mDet.det.sharpness || 100,
          lastSeen: now,
          opacity: 1.0,
          overlayElement: overlayEl
        };
        trackedFaces.set(newId, bestMatch);
      }

      matchedTrackIds.add(bestMatch.id);
      bestMatch.lastSeen = now;
      bestMatch.label = `Target #${mDet.targetIndex}`;
      bestMatch.target = { x: mDet.screenX, y: mDet.screenY, w: mDet.screenW, h: mDet.screenH };
      bestMatch.sharpness = mDet.det.sharpness || 100;

      // Temporal history buffer (5 frames)
      bestMatch.history.push({
        p_fake: mDet.det.p_fake || 0.0,
        verdict: mDet.det.verdict || "authentic",
        ts: now
      });
      if (bestMatch.history.length > 5) bestMatch.history.shift();

      // Rolling average of p_fake
      const avgScore = bestMatch.history.reduce((acc, h) => acc + h.p_fake, 0) / bestMatch.history.length;
      bestMatch.smoothedScore = avgScore;

      // Dual-Threshold Hysteresis:
      // Require 4 consecutive frames with p_fake >= 0.88 before declaring Deepfake
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

    // Immediately remove any tracked faces that were NOT in the current frame detection
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
      hudBadge.querySelector("#hv-badge-title").textContent = totalFaces > 1 ? `ANALYZING (${totalFaces} TARGETS)` : "ANALYZING (UNCERTAIN)";
      hudBadge.querySelector("#hv-badge-score").textContent = `Verifying Video... (${latency_ms || clientLatency}ms)`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = "Analyzing Video Consistency...";
      }
    } else {
      const minAuthConf = totalFaces > 0 ? Math.min(...activeList.map((f) => f.confidence)) : 95.0;
      hudBadge.className = "hv-hud-badge hv-authentic";
      hudBadge.querySelector("#hv-badge-title").textContent = totalFaces > 1 ? `VERIFIED (${totalFaces} TARGETS)` : "VERIFIED HUMAN";
      let bioText = `${minAuthConf}% Authentic`;
      if (data.bio_metrics && data.bio_metrics.blink_count !== undefined && data.bio_metrics.blink_count > 0) {
        bioText += ` • ${data.bio_metrics.blink_count} Blinks`;
      }
      hudBadge.querySelector("#hv-badge-score").textContent = `${bioText} (${latency_ms || clientLatency}ms)`;
      if (dockPill) {
        dockPill.querySelector("#hv-dock-sub").textContent = `${totalFaces} Face(s) Protected`;
      }
    }

    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: "DETECTION_UPDATE",
        payload: {
          verdict: suspiciousFaces.length > 0 ? "suspicious" : (uncertainFaces.length > 0 ? "uncertain" : "authentic"),
          confidence: totalFaces > 0 ? activeList[0].confidence : 95.0
        }
      }).catch(() => {});
    }
  }

  // Serialized Native Face Detection Loop (Strictly Non-Piling, 0% CPU accumulation)
  async function runNativeDetectionLoop() {
    if (!config.enabled || !nativeFaceDetector || !activeVideo || activeVideo.paused || activeVideo.ended || activeVideo.readyState < 2) {
      setTimeout(runNativeDetectionLoop, 150);
      return;
    }

    if (isNativeDetecting) return;
    isNativeDetecting = true;

    try {
      const detections = await nativeFaceDetector.detect(activeVideo);
      if (detections && detections.length > 0 && trackedFaces.size > 0) {
        const mirrored = isVideoMirrored(activeVideo);
        const geom = getRenderedVideoGeometry(activeVideo, activeVideo.videoWidth, activeVideo.videoHeight);
        const vw = activeVideo.videoWidth;
        const vh = activeVideo.videoHeight;

        detections.forEach((d) => {
          const b = d.boundingBox;
          const bx1 = b.x;
          const by1 = b.y;
          const bx2 = b.x + b.width;
          const by2 = b.y + b.height;

          const effX1 = mirrored ? (vw - bx2) : bx1;
          const effX2 = mirrored ? (vw - bx1) : bx2;

          const screenX = geom.left + geom.offsetX + (effX1 / vw) * geom.renderW;
          const screenY = geom.top + geom.offsetY + (by1 / vh) * geom.renderH;
          const screenW = ((effX2 - effX1) / vw) * geom.renderW;
          const screenH = ((by2 - by1) / vh) * geom.renderH;
          const cX = screenX + screenW / 2;
          const cY = screenY + screenH / 2;

          // Update closest tracked face immediately
          for (const face of trackedFaces.values()) {
            const fCX = face.current.x + face.current.w / 2;
            const fCY = face.current.y + face.current.h / 2;
            if (trackedFaces.size === 1 || Math.hypot(fCX - cX, fCY - cY) < 200) {
              face.target.x = screenX;
              face.target.y = screenY;
              face.target.w = screenW;
              face.target.h = screenH;
              break;
            }
          }
        });
      }
    } catch {
      // ignore
    } finally {
      isNativeDetecting = false;
      setTimeout(runNativeDetectionLoop, 80);
    }
  }

  if (nativeFaceDetector) {
    runNativeDetectionLoop();
  }

  // 60 FPS Adaptive Snap Lerp (Snaps in 1-2 frames when user moves their head)
  function animationLoop() {
    if (config.enabled && activeVideo && document.body.contains(activeVideo)) {
      updateHUDBadgePosition();

      for (const [id, face] of trackedFaces.entries()) {
        const dx = face.target.x - face.current.x;
        const dy = face.target.y - face.current.y;
        const dist = Math.hypot(dx, dy);

        // Snappy Velocity Lerp:
        // If distance > 60px: snap at 0.85 (catches up in 1-2 frames!)
        // If distance > 20px: snap at 0.60
        // Near rest (< 20px): ease at 0.35 to eliminate sensor jitter
        let lerpFactor = 0.35;
        if (dist > 60) {
          lerpFactor = 0.85;
        } else if (dist > 20) {
          lerpFactor = 0.60;
        }

        face.current.x += dx * lerpFactor;
        face.current.y += dy * lerpFactor;
        face.current.w += (face.target.w - face.current.w) * lerpFactor;
        face.current.h += (face.target.h - face.current.h) * lerpFactor;

        face.opacity = Math.min(1.0, face.opacity + 0.2);

        const el = face.overlayElement;
        if (el) {
          el.style.display = "block";
          el.style.left = `${Math.round(face.current.x)}px`;
          el.style.top = `${Math.round(face.current.y)}px`;
          el.style.width = `${Math.round(face.current.w)}px`;
          el.style.height = `${Math.round(face.current.h)}px`;
          el.style.opacity = face.opacity.toFixed(2);

          if (face.verdict === "suspicious") {
            el.className = "hv-face-overlay hv-suspicious-box";
            el.setAttribute("data-label", `${face.label}: ${face.confidence}% Risk (Deepfake)`);
          } else if (face.verdict === "uncertain") {
            el.className = "hv-face-overlay hv-uncertain-box";
            el.setAttribute("data-label", `${face.label}: Analyzing (${face.confidence}%)`);
          } else {
            el.className = "hv-face-overlay";
            el.setAttribute("data-label", `${face.label}: ${face.confidence}% Real`);
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
        <span class="hv-sub" id="hv-dock-sub">Multi-Face AR Shield Active</span>
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
      sub.textContent = "Multi-Face AR Shield Active";
    } else {
      btn.className = "hv-btn-toggle";
      btn.textContent = "Paused";
      label.textContent = "Shield Paused";
      sub.textContent = "Click to resume protection";
    }
  }

  // Debounced DOM Scanner (Only runs 1.0s after DOM stops changing)
  let domDebounce = null;
  const observer = new MutationObserver(() => {
    if (domDebounce) clearTimeout(domDebounce);
    domDebounce = setTimeout(() => {
      if (document.hidden) return;
      const v = findPrimaryVideo();
      if (v && v !== activeVideo) {
        activeVideo = v;
        attachHUD(v);
      }
    }, 1000);
  });

  observer.observe(document.body || document.documentElement, {
    childList: true,
    subtree: true
  });

  // Start 60 FPS animation loop
  animFrameId = requestAnimationFrame(animationLoop);

  // Dynamic Detection Loop (Guarded against overlapping calls)
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
