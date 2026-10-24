// Deepfake Shield — Real-Time Multi-Tile & Multi-Face Google Meet Protection
(function () {
  "use strict";

  // 1. NEVER inject into sub-frames or iframes
  if (window !== window.top) return;
  if (window.__HV_SHIELD_INJECTED__) return;
  window.__HV_SHIELD_INJECTED__ = true;

  // Runtime configuration
  // NOTE: mirrorMode is strictly PER-TILE and defaults to "auto" on every page load.
  // We intentionally do NOT persist a global mirrorMode override to avoid corrupting remote tiles.
  const config = {
    enabled: true,
    backendUrl: "http://127.0.0.1:8000",
    minTileScanIntervalMs: 500, // Cooldown between scans of the same tile
    globalScanTickMs: 120,      // Scheduler tick rate
    minGlobalIntervalMs: 250,   // Rate limit: max 4 requests/sec across all tiles combined
    debug: false
  };

  // Global state
  let overlayRoot = null;
  let dockPill = null;
  let schedulerRunning = false;
  let lastGlobalScanTs = 0;
  let tileIdCounter = 0;

  // Active Video Tile Contexts: Map<HTMLVideoElement, TileContext>
  const tileContexts = new Map();

  // Watchdog & mutation safety
  let writingOwnStyle = false;
  let globalChildObserver = null;
  let observerPaused = false;
  let observerCallCount = 0;
  let observerCallWindowStart = 0;

  // Rate-limited debug logging
  let lastDebugLogTs = 0;
  function debugLog(...args) {
    if (config.debug) {
      const now = performance.now();
      if (now - lastDebugLogTs > 2000) {
        lastDebugLogTs = now;
        console.log("%c[Deepfake Shield]", "color: #0ea5e9; font-weight: bold;", ...args);
      }
    }
  }

  // Developer console convenience: window.setDeepfakeShieldDebug(true/false)
  if (typeof window !== "undefined") {
    window.setDeepfakeShieldDebug = function (enabled) {
      config.debug = Boolean(enabled);
      console.log(`[Deepfake Shield] Debug logging set to: ${config.debug}`);
    };
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

  // Storage sync for enabled status and backendUrl
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["enabled", "backendUrl"], (stored) => {
      if (stored) {
        if (!stored.backendUrl || stored.backendUrl.includes("localhost") || stored.backendUrl.includes("onrender.com")) {
          stored.backendUrl = "http://127.0.0.1:8000";
        }
        if (stored.enabled !== undefined) {
          config.enabled = Boolean(stored.enabled);
        }
        config.backendUrl = stored.backendUrl;
      }
      updateDockPill();
    });

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local") {
        if (changes.enabled) {
          config.enabled = Boolean(changes.enabled.newValue);
          updateDockPill();
          if (!config.enabled) {
            hideAllOverlays();
          }
        }
        if (changes.backendUrl) {
          const v = changes.backendUrl.newValue;
          config.backendUrl = (!v || v.includes("localhost") || v.includes("onrender.com")) ? "http://127.0.0.1:8000" : v;
        }
      }
    });
  }

  // Shared Fixed Viewport Overlay Root (Immune to parent transforms & overflow clipping)
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

  // Safe style application (prevents MutationObserver feedback loops)
  function safeSetStyle(el, styles) {
    writingOwnStyle = true;
    for (const [k, v] of Object.entries(styles)) {
      el.style[k] = v;
    }
    Promise.resolve().then(() => {
      writingOwnStyle = false;
    });
  }

  // Lightweight 320p Frame Capture via Offscreen Canvas (~15KB JPEG)
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
    } catch {
      return null;
    }
  }

  // ==========================================
  // CSS Transform Matrix Analysis & Mirroring
  // ==========================================

  function parseTransform2D(transformStr) {
    if (!transformStr || transformStr === "none") return [1, 0, 0, 1];
    const t = transformStr.trim().toLowerCase();
    if (t.startsWith("matrix(")) {
      const parts = t.slice(7, -1).split(",").map((s) => parseFloat(s.trim()));
      if (parts.length >= 4 && !parts.slice(0, 4).some(isNaN)) {
        return [parts[0], parts[1], parts[2], parts[3]];
      }
    } else if (t.startsWith("matrix3d(")) {
      const parts = t.slice(9, -1).split(",").map((s) => parseFloat(s.trim()));
      if (parts.length >= 16 && !isNaN(parts[0]) && !isNaN(parts[1]) && !isNaN(parts[4]) && !isNaN(parts[5])) {
        return [parts[0], parts[1], parts[4], parts[5]];
      }
    }
    return [1, 0, 0, 1];
  }

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

  // Inspects the ancestor transform hierarchy of a SPECIFIC video element
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
          const id = `${el.tagName.toLowerCase()}${el.id ? "#" + el.id : ""}`;
          transforms.push(`${id}: ${t}`);
          chain.push(parseTransform2D(t));
        }
      } catch {}
      el = el.parentElement;
    }

    let cumulative = [1, 0, 0, 1];
    for (let i = chain.length - 1; i >= 0; i--) {
      cumulative = multiply2x2(cumulative, chain[i]);
    }

    const scaleX = cumulative[0];
    const isMirrored = scaleX < -0.001;

    return { isMirrored, scaleX, cumulative, transforms };
  }

  // Resolves the effective mirror state for a specific tile context
  function resolveTileMirrorState(tile) {
    const now = performance.now();
    // Cache auto-mirror computation for 1 second per tile
    if (now - tile.lastMirrorTs > 1000 || !tile.lastMirrorResult) {
      tile.lastMirrorResult = detectVideoMirroring(tile.video);
      tile.lastMirrorTs = now;
    }

    const autoMirrored = tile.lastMirrorResult.isMirrored;
    let shouldFlip = autoMirrored;
    let modeLabel = "auto";

    if (tile.mirrorMode === "flipped") {
      shouldFlip = true;
      modeLabel = "forced flipped";
    } else if (tile.mirrorMode === "normal") {
      shouldFlip = false;
      modeLabel = "forced normal";
    } else {
      shouldFlip = autoMirrored;
      modeLabel = "auto";
    }

    return { shouldFlip, modeLabel, mirrorInfo: tile.lastMirrorResult };
  }

  // ==========================================
  // Mathematical CSS object-fit Geometry
  // ==========================================

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
        renderH = elemH;
        renderW = elemH * videoRatio;
        offsetX = (elemW - renderW) / 2;
        offsetY = 0;
      } else {
        renderW = elemW;
        renderH = elemW / videoRatio;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      }
    } else {
      if (elemRatio > videoRatio) {
        renderW = elemW;
        renderH = elemW / videoRatio;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      } else {
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
      if (rect.width < 60 || rect.height < 50) return false;
      if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) return false;
      return true;
    } catch {
      return false;
    }
  }

  // ==========================================
  // Participant Name Extraction
  // ==========================================

  function extractParticipantName(video, tileIndex) {
    if (!video) return `Participant #${tileIndex}`;

    // 1. Direct self-view attributes
    const selfEl = video.closest("[data-self-name]");
    if (selfEl) {
      const name = selfEl.getAttribute("data-self-name");
      if (name && name.trim()) return `${name.trim()} (You)`;
      return "You";
    }

    if (
      video.closest("[data-is-self-view='true']") ||
      video.getAttribute("data-is-self-view") === "true" ||
      video.className.includes("self") ||
      (video.parentElement && video.parentElement.className.includes("self"))
    ) {
      return "You";
    }

    // 2. Search container hierarchy for participant badges/labels
    let container = video.parentElement;
    let depth = 0;
    while (container && depth < 8 && container !== document.body) {
      // Common Google Meet name selectors
      const nameEl = container.querySelector(
        ".XE8d1b, .poVWvd, [jsname='W297wb'], [jsname='A42nRe'], [data-self-name], [data-participant-id]"
      );
      if (nameEl && nameEl.textContent && nameEl.textContent.trim()) {
        const text = nameEl.textContent.trim();
        if (text.length > 0 && text.length < 35 && !text.includes(":") && !text.includes("Presenting")) {
          return text;
        }
      }

      // Aria label on tile container
      const aria = container.getAttribute("aria-label");
      if (aria && aria.trim() && !aria.includes("Google Meet") && !aria.includes("Call")) {
        return aria.trim();
      }

      container = container.parentElement;
      depth++;
    }

    return `Participant #${tileIndex}`;
  }

  // ==========================================
  // Multi-Tile Lifecycle Management
  // ==========================================

  function createTileContext(video, index) {
    tileIdCounter++;
    const tileId = `tile_${tileIdCounter}`;
    const participantName = extractParticipantName(video, index);
    const isSelfView = participantName.includes("You");

    // Per-tile HUD status badge
    const badge = document.createElement("div");
    badge.className = "hv-hud-badge hv-scanning";
    badge.id = `hv-badge-${tileId}`;
    badge.innerHTML = `
      <div class="hv-pulse-dot"></div>
      <div class="hv-hud-text">
        <span class="hv-badge-participant-name">${escapeHTML(participantName)}</span>
        <span class="hv-badge-title">PROTECTED</span>
        <span class="hv-hud-score hv-badge-score">Scanning...</span>
      </div>
      <button class="hv-badge-flip-btn" title="Toggle Mirror for this tile">⇄ Auto</button>
    `;

    document.body.appendChild(badge);

    const tile = {
      id: tileId,
      video,
      participantName,
      isSelfView,
      mirrorMode: "auto", // "auto" (default: CSS matrix), "flipped", "normal"
      lastMirrorResult: null,
      lastMirrorTs: 0,
      hudBadge: badge,
      trackedFaces: new Map(), // faceId -> TrackedFace
      inFlight: false,
      lastScanTs: 0,
      consecutiveNoFace: 0,
      lastVerdict: "searching",
      resizeObserver: null,
      videoObserver: null
    };

    // Wire up per-tile flip button
    const flipBtn = badge.querySelector(".hv-badge-flip-btn");
    if (flipBtn) {
      updateTileFlipButton(tile);
      flipBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (tile.mirrorMode === "auto") {
          tile.mirrorMode = "flipped";
        } else if (tile.mirrorMode === "flipped") {
          tile.mirrorMode = "normal";
        } else {
          tile.mirrorMode = "auto";
        }
        tile.lastMirrorTs = 0; // force recompute
        updateTileFlipButton(tile);
        repositionTileFaces(tile);
      });
    }

    // Attach ResizeObserver to video and direct parent
    if (window.ResizeObserver) {
      tile.resizeObserver = new ResizeObserver(() => {
        tile.lastMirrorTs = 0; // invalidate mirror cache on resize
        repositionTileFaces(tile);
        repositionTileBadge(tile);
      });
      tile.resizeObserver.observe(video);
      if (video.parentElement) {
        tile.resizeObserver.observe(video.parentElement);
      }
    }

    // Observe ONLY the <video> element itself (never parent trees) for style/class changes
    tile.videoObserver = new MutationObserver(() => {
      if (writingOwnStyle || observerPaused) return;

      // Watchdog tracking
      const now = performance.now();
      if (now - observerCallWindowStart > 1000) {
        observerCallCount = 0;
        observerCallWindowStart = now;
      }
      observerCallCount++;
      if (observerCallCount > 30) {
        observerPaused = true;
        console.warn("[Deepfake Shield] Observer activity spike (>30/s). Pausing tile observer for 3 seconds.");
        if (tile.videoObserver) tile.videoObserver.disconnect();
        setTimeout(() => {
          observerPaused = false;
          observerCallCount = 0;
          if (tile.video && document.body.contains(tile.video)) {
            tile.videoObserver.observe(tile.video, { attributes: true, attributeFilter: ["style", "class"] });
          }
        }, 3000);
        return;
      }

      tile.lastMirrorTs = 0;
      repositionTileFaces(tile);
      repositionTileBadge(tile);
    });

    tile.videoObserver.observe(video, { attributes: true, attributeFilter: ["style", "class"] });

    repositionTileBadge(tile);
    return tile;
  }

  function updateTileFlipButton(tile) {
    if (!tile.hudBadge) return;
    const btn = tile.hudBadge.querySelector(".hv-badge-flip-btn");
    if (!btn) return;

    const { shouldFlip } = resolveTileMirrorState(tile);

    let text, borderColor;
    if (tile.mirrorMode === "auto") {
      text = shouldFlip ? "⇄ Auto (Mirrored)" : "⇄ Auto (Normal)";
      borderColor = "rgba(255, 255, 255, 0.35)";
    } else if (tile.mirrorMode === "flipped") {
      text = "⇄ Force Flipped";
      borderColor = "rgba(245, 158, 11, 0.85)";
    } else {
      text = "⇄ Force Normal";
      borderColor = "rgba(59, 130, 246, 0.85)";
    }

    if (btn.textContent !== text) btn.textContent = text;
    if (btn.style.borderColor !== borderColor) {
      safeSetStyle(btn, { borderColor });
    }
  }

  function destroyTileContext(tile) {
    if (tile.resizeObserver) {
      tile.resizeObserver.disconnect();
      tile.resizeObserver = null;
    }
    if (tile.videoObserver) {
      tile.videoObserver.disconnect();
      tile.videoObserver = null;
    }
    if (tile.hudBadge && tile.hudBadge.parentNode) {
      tile.hudBadge.parentNode.removeChild(tile.hudBadge);
    }
    // Remove all face overlays for this tile
    for (const [, face] of tile.trackedFaces) {
      if (face.overlayElement && face.overlayElement.parentNode) {
        face.overlayElement.parentNode.removeChild(face.overlayElement);
      }
    }
    tile.trackedFaces.clear();
  }

  function scanAndSyncActiveTiles() {
    if (!config.enabled || document.hidden) return;

    const allVideos = Array.from(document.querySelectorAll("video"));
    const activeCandidates = allVideos.filter((v) => {
      if (v.paused || v.ended || v.readyState < 2) return false;
      if (!v.videoWidth || !v.videoHeight || v.videoWidth < 40 || v.videoHeight < 40) return false;
      return isElementVisible(v);
    });

    // 1. Remove tiles whose videos no longer exist or are hidden
    for (const [video, tile] of tileContexts.entries()) {
      if (!activeCandidates.includes(video) || !document.body.contains(video)) {
        destroyTileContext(tile);
        tileContexts.delete(video);
      }
    }

    // 2. Add newly discovered video tiles
    activeCandidates.forEach((v, idx) => {
      if (!tileContexts.has(v)) {
        const tile = createTileContext(v, idx + 1);
        tileContexts.set(v, tile);
      } else {
        // Refresh participant name if it updated
        const existingTile = tileContexts.get(v);
        const curName = extractParticipantName(v, idx + 1);
        if (existingTile.participantName !== curName) {
          existingTile.participantName = curName;
          const nameSpan = existingTile.hudBadge.querySelector(".hv-badge-participant-name");
          if (nameSpan) nameSpan.textContent = curName;
        }
      }
    });

    // 3. Update HUD badge positions
    for (const [, tile] of tileContexts) {
      repositionTileBadge(tile);
    }
  }

  function repositionTileBadge(tile) {
    if (!tile.hudBadge || !tile.video || !document.body.contains(tile.video)) return;
    const rect = tile.video.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      safeSetStyle(tile.hudBadge, { display: "none" });
      return;
    }
    safeSetStyle(tile.hudBadge, {
      display: "flex",
      left: `${Math.round(rect.left + 14)}px`,
      top: `${Math.round(rect.top + 14)}px`
    });
  }

  function hideAllOverlays() {
    for (const [, tile] of tileContexts) {
      if (tile.hudBadge) safeSetStyle(tile.hudBadge, { display: "none" });
      for (const [, face] of tile.trackedFaces) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
      }
      tile.trackedFaces.clear();
    }
  }

  // =========================================================================
  // Multi-Face IoU & Center-Distance Matching (Prevents ID Swapping)
  // =========================================================================

  function boxIoU(b1, b2) {
    const xA = Math.max(b1[0], b2[0]);
    const yA = Math.max(b1[1], b2[1]);
    const xB = Math.min(b1[0] + b1[2], b2[0] + b2[2]);
    const yB = Math.min(b1[1] + b1[3], b2[1] + b2[3]);
    const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (inter <= 0) return 0.0;
    const area1 = b1[2] * b1[3];
    const area2 = b2[2] * b2[3];
    return inter / Math.max(1e-5, area1 + area2 - inter);
  }

  function centerDistance(b1, b2) {
    const c1x = b1[0] + b1[2] / 2;
    const c1y = b1[1] + b1[3] / 2;
    const c2x = b2[0] + b2[2] / 2;
    const c2y = b2[1] + b2[3] / 2;
    const dx = c1x - c2x;
    const dy = c1y - c2y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Repositions all face bounding boxes for a specific tile based on current geometry
  function repositionTileFaces(tile) {
    if (!tile.video || tile.trackedFaces.size === 0) return;
    const geom = getRenderedVideoGeometry(tile.video);
    if (!geom) return;

    const { shouldFlip } = resolveTileMirrorState(tile);

    for (const [, face] of tile.trackedFaces) {
      if (!face.lastNormBox || !face.overlayElement) continue;

      const [x_norm, y_norm, w_norm, h_norm] = face.lastNormBox;
      const screen_x_norm = shouldFlip ? (1.0 - (x_norm + w_norm)) : x_norm;

      const targetX = geom.left + geom.offsetX + (screen_x_norm * geom.renderW);
      const targetY = geom.top + geom.offsetY + (y_norm * geom.renderH);
      const targetW = w_norm * geom.renderW;
      const targetH = h_norm * geom.renderH;

      face.smoothed = { x: targetX, y: targetY, w: targetW, h: targetH };

      safeSetStyle(face.overlayElement, {
        left: `${Math.round(targetX)}px`,
        top: `${Math.round(targetY)}px`,
        width: `${Math.round(targetW)}px`,
        height: `${Math.round(targetH)}px`
      });
    }
  }

  // =========================================================================
  // Tile Inference Processing & Per-Face Overlay Rendering
  // =========================================================================

  function processTileDetections(tile, data, capture, clientLatency) {
    const root = getOrCreateOverlayRoot();
    const video = tile.video;
    const badge = tile.hudBadge;
    if (!video || !badge) return;

    const { verdict, face_detected, latency_ms, emotion } = data;
    const geom = getRenderedVideoGeometry(video);
    if (!geom) return;

    const fw = data.frame_width || capture.capturedW || 320;
    const fh = data.frame_height || capture.capturedH || 180;

    const { shouldFlip, modeLabel } = resolveTileMirrorState(tile);
    updateTileFlipButton(tile);

    // ZERO FACES DETECTED IN THIS TILE: Purge ghost boxes & set searching state
    if (!face_detected || verdict === "no_face" || !data.faces || data.faces.length === 0) {
      tile.consecutiveNoFace++;
      tile.lastVerdict = "no_face";

      badge.className = "hv-hud-badge hv-no-face";
      const titleSpan = badge.querySelector(".hv-badge-title");
      const scoreSpan = badge.querySelector(".hv-badge-score");
      if (titleSpan) titleSpan.textContent = "NO FACE";
      if (scoreSpan) scoreSpan.textContent = "Scanning...";

      // Clean up any stale face overlays for this tile
      for (const [, face] of tile.trackedFaces) {
        if (face.overlayElement && face.overlayElement.parentNode) {
          face.overlayElement.parentNode.removeChild(face.overlayElement);
        }
      }
      tile.trackedFaces.clear();
      repositionTileBadge(tile);
      return;
    }

    tile.consecutiveNoFace = 0;
    const rawFaces = data.faces;

    // Prepare normalized candidates
    const candidates = rawFaces.map((det) => {
      const box = det.bounding_box || [0, 0, 0, 0];

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

      // Exact horizontal screen coordinate flip
      const screen_x_norm = shouldFlip ? (1.0 - (x_norm + w_norm)) : x_norm;

      const targetX = geom.left + geom.offsetX + (screen_x_norm * geom.renderW);
      const targetY = geom.top + geom.offsetY + (y_norm * geom.renderH);
      const targetW = w_norm * geom.renderW;
      const targetH = h_norm * geom.renderH;
      const centerX = targetX + targetW / 2;
      const centerY = targetY + targetH / 2;

      return {
        det,
        targetX, targetY, targetW, targetH,
        centerX, centerY,
        rawNormBox
      };
    });

    // Pairwise matching with existing tracked faces
    const existingFaces = Array.from(tile.trackedFaces.values());
    const matchedExistingIds = new Set();
    const matchedCandidateIndices = new Set();

    // Compute pairwise similarity matrix
    const matchPairs = [];
    existingFaces.forEach((f) => {
      candidates.forEach((c, cIdx) => {
        const iou = boxIoU(f.lastNormBox, c.rawNormBox);
        const dist = centerDistance(f.lastNormBox, c.rawNormBox);
        // Higher score = better match
        let score = 0;
        if (iou > 0.15) {
          score = 10.0 + iou;
        } else if (dist < 0.35) {
          score = 5.0 + (1.0 - dist);
        }
        if (score > 0) {
          matchPairs.push({ face: f, cIdx, score });
        }
      });
    });

    // Greedily match best scoring pairs
    matchPairs.sort((a, b) => b.score - a.score);
    matchPairs.forEach((pair) => {
      if (!matchedExistingIds.has(pair.face.id) && !matchedCandidateIndices.has(pair.cIdx)) {
        matchedExistingIds.add(pair.face.id);
        matchedCandidateIndices.add(pair.cIdx);

        // Update matched face
        const c = candidates[pair.cIdx];
        updateTrackedFace(pair.face, c, geom);
      }
    });

    // Create new TrackedFace for unmatched detections
    candidates.forEach((c, cIdx) => {
      if (!matchedCandidateIndices.has(cIdx)) {
        const newFace = createTrackedFace(tile, c, root, geom);
        updateTrackedFace(newFace, c, geom);
      }
    });

    // Remove stale faces not matched in this cycle
    for (const [id, face] of tile.trackedFaces) {
      if (!matchedExistingIds.has(id)) {
        face.missedCount = (face.missedCount || 0) + 1;
        if (face.missedCount >= 2) {
          if (face.overlayElement && face.overlayElement.parentNode) {
            face.overlayElement.parentNode.removeChild(face.overlayElement);
          }
          tile.trackedFaces.delete(id);
        }
      } else {
        face.missedCount = 0;
      }
    }

    repositionTileBadge(tile);

    // Update HUD Badge Status for this specific tile
    const activeFaces = Array.from(tile.trackedFaces.values());
    const totalFaces = activeFaces.length;
    const suspiciousFaces = activeFaces.filter((f) => f.verdict === "suspicious");
    const primaryEmotion = (activeFaces.length > 0 && activeFaces[0].emotion) || emotion || "Focused / Calm 👤";

    const titleSpan = badge.querySelector(".hv-badge-title");
    const scoreSpan = badge.querySelector(".hv-badge-score");

    if (suspiciousFaces.length > 0) {
      tile.lastVerdict = "suspicious";
      const maxRisk = Math.max(...suspiciousFaces.map((f) => f.confidence));
      badge.className = "hv-hud-badge hv-suspicious";
      if (titleSpan) titleSpan.textContent = totalFaces > 1 ? `DEEPFAKE ALERT (${suspiciousFaces.length}/${totalFaces})` : "DEEPFAKE ALERT";
      if (scoreSpan) scoreSpan.textContent = `${maxRisk}% Risk (${latency_ms || clientLatency}ms)`;
    } else {
      tile.lastVerdict = "authentic";
      const minAuthConf = totalFaces > 0 ? Math.min(...activeFaces.map((f) => f.confidence)) : 99.0;
      badge.className = "hv-hud-badge hv-authentic";
      if (titleSpan) titleSpan.textContent = "VERIFIED";
      if (scoreSpan) scoreSpan.textContent = `${minAuthConf}% Authentic • ${primaryEmotion}`;
    }

    debugLog(
      `[${tile.participantName}] faces=${totalFaces} flip=${shouldFlip}(${modeLabel}) verdict=${tile.lastVerdict}`
    );
  }

  function createTrackedFace(tile, candidate, root, geom) {
    const existingIndices = Array.from(tile.trackedFaces.values()).map((f) => f.index);
    let targetIndex = 1;
    while (existingIndices.includes(targetIndex)) targetIndex++;

    const faceId = `${tile.id}_face_${targetIndex}`;

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

    const face = {
      id: faceId,
      index: targetIndex,
      label: `Face #${targetIndex}`,
      overlayElement: overlayEl,
      tagElement: tagEl,
      crosshairElement: crosshairEl,
      lastNormBox: candidate.rawNormBox,
      smoothed: { x: candidate.targetX, y: candidate.targetY, w: candidate.targetW, h: candidate.targetH },
      verdict: "authentic",
      confidence: 100.0,
      emotion: "Focused / Calm 👤",
      missedCount: 0
    };

    tile.trackedFaces.set(faceId, face);
    return face;
  }

  function updateTrackedFace(face, candidate, geom) {
    face.lastNormBox = candidate.rawNormBox;
    face.emotion = candidate.det.emotion || "Focused / Calm 👤";
    face.verdict = candidate.det.verdict || "authentic";
    face.confidence = candidate.det.confidence || 100.0;

    // Relative displacement of center for dynamic EMA snapping
    const prevCenterX = face.smoothed.x + face.smoothed.w / 2;
    const prevCenterY = face.smoothed.y + face.smoothed.h / 2;
    const relDx = Math.abs(candidate.centerX - prevCenterX) / Math.max(1, geom.renderW);
    const relDy = Math.abs(candidate.centerY - prevCenterY) / Math.max(1, geom.renderH);
    const relDist = Math.sqrt(relDx * relDx + relDy * relDy);

    const isFastMovement = relDist > 0.10;
    const alpha = isFastMovement ? 1.0 : 0.70;

    face.smoothed.x = alpha * candidate.targetX + (1 - alpha) * face.smoothed.x;
    face.smoothed.y = alpha * candidate.targetY + (1 - alpha) * face.smoothed.y;
    face.smoothed.w = alpha * candidate.targetW + (1 - alpha) * face.smoothed.w;
    face.smoothed.h = alpha * candidate.targetH + (1 - alpha) * face.smoothed.h;

    const el = face.overlayElement;
    if (el) {
      const transition = isFastMovement
        ? "none"
        : "left 0.08s ease-out, top 0.08s ease-out, width 0.08s ease-out, height 0.08s ease-out, border-color 0.25s, box-shadow 0.25s, opacity 0.25s";

      safeSetStyle(el, {
        transition,
        display: "block",
        left: `${Math.round(face.smoothed.x)}px`,
        top: `${Math.round(face.smoothed.y)}px`,
        width: `${Math.round(face.smoothed.w)}px`,
        height: `${Math.round(face.smoothed.h)}px`,
        opacity: "1"
      });

      const label = face.label;
      const emotionText = face.emotion;

      if (face.verdict === "suspicious") {
        el.className = "hv-face-overlay hv-suspicious-box";
        if (face.tagElement) {
          face.tagElement.innerHTML = `
            <span class="hv-tag-identity">${escapeHTML(label)}</span>
            <span class="hv-tag-verdict hv-v-susp">${face.confidence}% Risk (Deepfake)</span>
            <span class="hv-tag-emotion">${escapeHTML(emotionText)}</span>
          `;
        }
      } else {
        el.className = "hv-face-overlay";
        if (face.tagElement) {
          face.tagElement.innerHTML = `
            <span class="hv-tag-identity">${escapeHTML(label)}</span>
            <span class="hv-tag-verdict hv-v-auth">${face.confidence}% Real</span>
            <span class="hv-tag-emotion">${escapeHTML(emotionText)}</span>
          `;
        }
      }
    }
  }

  // =========================================================================
  // Fair Priority & Round-Robin Tile Scheduler (Bounded Load, Zero IPC Flood)
  // =========================================================================

  function selectNextScheduledTile() {
    const now = performance.now();
    const candidates = [];

    for (const [, tile] of tileContexts) {
      // Must not have an in-flight request
      if (tile.inFlight) continue;

      // Check per-tile minimum scan cooldown
      if (now - tile.lastScanTs < config.minTileScanIntervalMs) continue;

      const video = tile.video;
      if (!video || !document.body.contains(video) || video.paused || video.ended || video.readyState < 2) continue;

      const rect = video.getBoundingClientRect();
      const area = rect.width * rect.height;
      if (area < 2000) continue; // Ignore microscopic elements

      // Priority scoring:
      // 1. Larger tiles (spotlight / pinned / active speaker)
      // 2. Self-view
      // 3. Tile already holding tracked faces
      // 4. Age since last scan
      let score = area / 10000;
      if (tile.isSelfView) score += 15.0;
      if (tile.trackedFaces.size > 0) score += 25.0;
      score += (now - tile.lastScanTs) / 500.0;

      candidates.push({ tile, score });
    }

    if (candidates.length === 0) return null;
    candidates.sort((a, b) => b.score - a.score);
    return candidates[0].tile;
  }

  async function scheduleScanCycle() {
    if (!config.enabled || document.hidden) return;

    const now = performance.now();
    if (now - lastGlobalScanTs < config.minGlobalIntervalMs) return;

    const tile = selectNextScheduledTile();
    if (!tile) return;

    const capture = captureCompactFrame(tile.video);
    if (!capture) return;

    tile.inFlight = true;
    lastGlobalScanTs = now;
    const reqStart = performance.now();

    const watchdogTimer = setTimeout(() => {
      if (tile.inFlight) {
        tile.inFlight = false;
        tile.lastScanTs = performance.now();
      }
    }, 6000);

    const { shouldFlip } = resolveTileMirrorState(tile);

    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      try {
        chrome.runtime.sendMessage(
          {
            type: "PREDICT_FRAME",
            payload: {
              backendUrl: config.backendUrl,
              image_base64: capture.dataUrl,
              mirrored: shouldFlip
            }
          },
          (response) => {
            clearTimeout(watchdogTimer);
            tile.inFlight = false;
            tile.lastScanTs = performance.now();

            if (chrome.runtime.lastError || !response || !response.success) {
              return;
            }

            const clientLatency = Math.round(performance.now() - reqStart);
            try {
              processTileDetections(tile, response.data, capture, clientLatency);
            } catch (err) {
              debugLog("Tile detection processing error:", err);
            }
          }
        );
      } catch {
        clearTimeout(watchdogTimer);
        tile.inFlight = false;
        tile.lastScanTs = performance.now();
      }
    } else {
      clearTimeout(watchdogTimer);
      tile.inFlight = false;
    }
  }

  async function startSchedulerLoop() {
    if (schedulerRunning) return;
    schedulerRunning = true;

    while (true) {
      try {
        scanAndSyncActiveTiles();
        await scheduleScanCycle();
      } catch (err) {
        debugLog("Scheduler error:", err);
      }
      await sleep(config.globalScanTickMs);
    }
  }

  // ==========================================
  // Draggable HUD Dock Pill
  // ==========================================

  function createDockPill() {
    if (dockPill || document.getElementById("hv-dock-pill")) return;

    dockPill = document.createElement("div");
    dockPill.id = "hv-dock-pill";
    dockPill.className = "hv-dock-pill";

    dockPill.innerHTML = `
      <div class="hv-drag-handle" title="Drag to move shield">⋮⋮</div>
      <div class="hv-shield-icon">
        <svg viewBox="0 0 24 24">
          <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-1 6h2v6h-2V7zm0 8h2v2h-2v-2z"/>
        </svg>
      </div>
      <div class="hv-status-label">
        <span id="hv-dock-text">Deepfake Shield Active</span>
        <span class="hv-sub" id="hv-dock-sub">Multi-Participant Video &amp; Emotion Shield</span>
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
      if (!config.enabled) {
        hideAllOverlays();
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

      safeSetStyle(element, {
        bottom: "auto",
        right: "auto",
        left: `${initialLeft}px`,
        top: `${initialTop}px`
      });

      e.preventDefault();
    }

    function onPointerMove(e) {
      if (!isDragging) return;
      const clientX = e.clientX || (e.touches && e.touches[0].clientX);
      const clientY = e.clientY || (e.touches && e.touches[0].clientY);

      let newLeft = initialLeft + (clientX - startX);
      let newTop = initialTop + (clientY - startY);

      const maxLeft = window.innerWidth - element.offsetWidth - 10;
      const maxTop = window.innerHeight - element.offsetHeight - 10;
      newLeft = Math.max(10, Math.min(newLeft, maxLeft));
      newTop = Math.max(10, Math.min(newTop, maxTop));

      safeSetStyle(element, {
        left: `${newLeft}px`,
        top: `${newTop}px`
      });
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
      sub.textContent = "Multi-Participant Video & Emotion Shield";
    } else {
      btn.className = "hv-btn-toggle";
      btn.textContent = "Paused";
      label.textContent = "Shield Paused";
      sub.textContent = "Click to resume protection";
    }
  }

  // Window resize handler
  window.addEventListener("resize", () => {
    for (const [, tile] of tileContexts) {
      repositionTileFaces(tile);
      repositionTileBadge(tile);
    }
  });

  // Global DOM childList observer (Watches ONLY additions/removals of DOM nodes, NEVER style or class)
  globalChildObserver = new MutationObserver(() => {
    if (document.hidden || !config.enabled) return;
    scanAndSyncActiveTiles();
  });
  globalChildObserver.observe(document.body, { childList: true, subtree: true });

  // Initialize
  startSchedulerLoop();
  createDockPill();
})();
