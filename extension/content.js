// Deepfake Shield — Unified Real-Time Google Meet & Video Call Protection
// Built with unified shared/boxMapping.js engine & 60 FPS AF-LOCK reticle renderer
(function () {
  "use strict";

  // Prevent multiple injections
  if (window !== window.top) return;
  if (window.__HV_SHIELD_INJECTED__) return;
  window.__HV_SHIELD_INJECTED__ = true;

  // Runtime configuration
  const config = {
    enabled: true,
    backendUrl: "http://127.0.0.1:8000",
    minTileScanIntervalMs: 450, // Per-tile scan cooldown
    globalScanTickMs: 100,      // Scheduler tick rate
    minGlobalIntervalMs: 220,   // Rate limit: max 4.5 req/sec across all tiles combined
    debug: false
  };

  // Global state
  let overlayRoot = null;
  let hudCanvas = null;
  let hudCtx = null;
  let hudAnimId = null;
  let dockPill = null;
  let schedulerRunning = false;
  let lastGlobalScanTs = 0;
  let tileIdCounter = 0;
  let lastBackendResponse = null;
  let lastError = null;

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
      if (now - lastDebugLogTs > 1500) {
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

  // Shared Box Mapping Engine: Access unified shared/boxMapping.js with built-in fallback
  const boxEngine = (typeof window !== "undefined" && window.DeepfakeShieldBoxMapping) ? window.DeepfakeShieldBoxMapping : {
    computeRenderGeometry: function (elemW, elemH, videoW, videoH, objectFit = "cover") {
      const vw = videoW || elemW || 640;
      const vh = videoH || elemH || 480;
      if (elemW <= 0 || elemH <= 0 || vw <= 0 || vh <= 0) return { renderW: elemW, renderH: elemH, offsetX: 0, offsetY: 0 };
      const videoRatio = vw / vh;
      const elemRatio = elemW / elemH;
      let renderW = elemW;
      let renderH = elemH;
      let offsetX = 0;
      let offsetY = 0;
      if (objectFit === "contain") {
        if (elemRatio > videoRatio) {
          renderH = elemH; renderW = elemH * videoRatio; offsetX = (elemW - renderW) / 2;
        } else {
          renderW = elemW; renderH = elemW / videoRatio; offsetY = (elemH - renderH) / 2;
        }
      } else {
        if (elemRatio > videoRatio) {
          renderW = elemW; renderH = elemW / videoRatio; offsetY = (elemH - renderH) / 2;
        } else {
          renderH = elemH; renderW = elemH * videoRatio; offsetX = (elemW - renderW) / 2;
        }
      }
      return { renderW, renderH, offsetX, offsetY };
    },
    mapFaceToScreen: function (face, frameW, frameH, videoRect, videoW, videoH, objectFit = "cover", isMirrored = false) {
      const fw = frameW || 480;
      const fh = frameH || 360;
      const elemW = videoRect.width || fw;
      const elemH = videoRect.height || fh;
      let normX, normY, normW, normH;
      if (face.norm_xywh && face.norm_xywh.length >= 4) {
        [normX, normY, normW, normH] = face.norm_xywh;
      } else if (face.bounding_box && face.bounding_box.length >= 4) {
        const [bx1, by1, bx2, by2] = face.bounding_box;
        normX = bx1 / fw; normY = by1 / fh; normW = (bx2 - bx1) / fw; normH = (by2 - by1) / fh;
      } else {
        normX = 0; normY = 0; normW = 0.5; normH = 0.5;
      }
      normX = Math.max(0.0, Math.min(1.0, normX));
      normY = Math.max(0.0, Math.min(1.0, normY));
      normW = Math.max(0.01, Math.min(1.0 - normX, normW));
      normH = Math.max(0.01, Math.min(1.0 - normY, normH));
      const geom = this.computeRenderGeometry(elemW, elemH, videoW, videoH, objectFit);
      const effNormX = isMirrored ? (1.0 - (normX + normW)) : normX;
      const localX = geom.offsetX + (effNormX * geom.renderW);
      const localY = geom.offsetY + (normY * geom.renderH);
      const boxW = normW * geom.renderW;
      const boxH = normH * geom.renderH;
      const screenX = (videoRect.left || 0) + localX;
      const screenY = (videoRect.top || 0) + localY;
      return { screenX, screenY, width: boxW, height: boxH, centerX: screenX + boxW / 2, centerY: screenY + boxH / 2, localX, localY, normBox: [normX, normY, normW, normH] };
    },
    drawAFLockReticle: function (ctx, bx1, by1, bx2, by2, face, targetIndex = 1, opacity = 1.0, canvasW = window.innerWidth) {
      const bw = Math.max(20, bx2 - bx1);
      const bh = Math.max(20, by2 - by1);
      const cx = bx1 + bw / 2;
      const cy = by1 + bh / 2;
      const cornerLen = Math.max(14, Math.min(bw, bh) * 0.22);
      const isFake = face.verdict === "suspicious" || (face.p_fake || 0) >= 0.65;
      const strokeColor = isFake ? "#ef4444" : "#10b981";
      ctx.save();
      ctx.globalAlpha = Math.max(0.1, Math.min(1.0, opacity));
      ctx.shadowColor = strokeColor;
      ctx.shadowBlur = 10;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(bx1, by1 + cornerLen); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + cornerLen, by1);
      ctx.moveTo(bx2 - cornerLen, by1); ctx.lineTo(bx2, by1); ctx.lineTo(bx2, by1 + cornerLen);
      ctx.moveTo(bx1, by2 - cornerLen); ctx.lineTo(bx1, by2); ctx.lineTo(bx1 + cornerLen, by2);
      ctx.moveTo(bx2 - cornerLen, by2); ctx.lineTo(bx2, by2); ctx.lineTo(bx2, by2 - cornerLen);
      ctx.stroke();
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
      ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
      ctx.stroke();
      const badgeText = isFake ? `🔴 AF-ALERT #${targetIndex}: DEEPFAKE ${face.confidence || 90}%` : `🟢 AF-LOCK #${targetIndex}: AUTHENTIC ${face.confidence || 99}%`;
      const emotionText = face.emotion ? String(face.emotion) : "Focused / Calm 👤";
      ctx.font = "bold 11px Inter, system-ui, -apple-system, sans-serif";
      const textW = Math.max(ctx.measureText(badgeText).width, ctx.measureText(emotionText).width);
      const badgeH = 34;
      const badgeX = Math.max(8, Math.min(canvasW - textW - 14, cx - textW / 2));
      const badgeY = Math.max(badgeH + 4, by1 - 8);
      ctx.fillStyle = "rgba(15, 23, 42, 0.94)";
      ctx.fillRect(badgeX - 4, badgeY - badgeH + 4, textW + 12, badgeH);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(badgeX - 4, badgeY - badgeH + 4, textW + 12, badgeH);
      ctx.fillStyle = strokeColor;
      ctx.fillText(badgeText, badgeX + 2, badgeY - 14);
      ctx.font = "600 10px Inter, system-ui, -apple-system, sans-serif";
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(emotionText, badgeX + 2, badgeY);
      ctx.restore();
    }
  };

  // Fixed Viewport Overlay Root and 60 FPS Canvas Reticle Overlay
  function getOrCreateOverlayRoot() {
    if (overlayRoot && document.body.contains(overlayRoot)) return overlayRoot;
    let root = document.getElementById("hv-overlay-root");
    if (!root) {
      root = document.createElement("div");
      root.id = "hv-overlay-root";
      document.body.appendChild(root);
    }
    overlayRoot = root;

    // Viewport-wide 60 FPS HUD Canvas
    if (!hudCanvas || !document.body.contains(hudCanvas)) {
      hudCanvas = document.getElementById("hv-hud-canvas");
      if (!hudCanvas) {
        hudCanvas = document.createElement("canvas");
        hudCanvas.id = "hv-hud-canvas";
        hudCanvas.style.cssText = "position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483642;display:block;";
        overlayRoot.appendChild(hudCanvas);
      }
      hudCtx = hudCanvas.getContext("2d");
      resizeHUDCanvas();
    }

    return overlayRoot;
  }

  function resizeHUDCanvas() {
    if (!hudCanvas) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (hudCanvas.width !== w || hudCanvas.height !== h) {
      hudCanvas.width = w;
      hudCanvas.height = h;
    }
  }

  window.addEventListener("resize", () => {
    resizeHUDCanvas();
    for (const [, tile] of tileContexts) {
      repositionTileBadge(tile);
    }
  });

  // Safe style application without triggering feedback loops
  function safeSetStyle(el, styles) {
    writingOwnStyle = true;
    for (const [k, v] of Object.entries(styles)) {
      el.style[k] = v;
    }
    Promise.resolve().then(() => {
      writingOwnStyle = false;
    });
  }

  // Lightweight Frame Capture via Offscreen Canvas
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  function captureCompactFrame(video) {
    if (!video || video.paused || video.ended) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || vw < 20 || vh < 20) return null;

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
        dataUrl: offscreenCanvas.toDataURL("image/jpeg", 0.72),
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

  function resolveTileMirrorState(tile) {
    const now = performance.now();
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
  // Participant Name Extraction Pipeline
  // ==========================================

  function extractParticipantName(video, tileIndex) {
    if (!video) return `Participant #${tileIndex}`;

    function cleanName(raw, isSelf) {
      if (!raw) return isSelf ? "You" : `Participant #${tileIndex}`;
      let text = String(raw).trim();
      text = text.replace(/\s*\(\s*You\s*\)\s*/gi, "").trim();
      if (!text || text.toLowerCase() === "you") {
        return "You";
      }
      return isSelf ? `${text} (You)` : text;
    }

    // 1. Check self-view indicator
    const selfEl = video.closest("[data-self-name]");
    if (selfEl) {
      const name = selfEl.getAttribute("data-self-name");
      return cleanName(name, true);
    }

    if (
      video.closest("[data-is-self-view='true']") ||
      video.getAttribute("data-is-self-view") === "true" ||
      video.className.includes("self") ||
      (video.parentElement && video.parentElement.className.includes("self"))
    ) {
      return "You";
    }

    // 2. Search container hierarchy
    let container = video.parentElement;
    let depth = 0;
    while (container && depth < 8 && container !== document.body) {
      const nameEl = container.querySelector(
        ".XE8d1b, .poVWvd, [jsname='W297wb'], [jsname='A42nRe'], [data-self-name], [data-participant-id]"
      );
      if (nameEl && nameEl.textContent && nameEl.textContent.trim()) {
        const text = nameEl.textContent.trim();
        if (text.length > 0 && text.length < 35 && !text.includes(":") && !text.includes("Presenting")) {
          return cleanName(text, false);
        }
      }

      const aria = container.getAttribute("aria-label");
      if (aria && aria.trim() && !aria.includes("Google Meet") && !aria.includes("Call")) {
        return cleanName(aria, false);
      }

      container = container.parentElement;
      depth++;
    }

    return `Participant #${tileIndex}`;
  }

  // ==========================================
  // Video Discovery & Validation Evaluator
  // ==========================================

  function evaluateVideoCandidate(v) {
    if (!v) return { accepted: false, reason: "null video element" };
    if (!document.body.contains(v)) return { accepted: false, reason: "detached from document" };
    if (v.paused || v.ended) return { accepted: false, reason: "paused or ended stream" };

    const rect = v.getBoundingClientRect();
    const vw = v.videoWidth || 0;
    const vh = v.videoHeight || 0;

    if (rect.width < 40 || rect.height < 40) {
      return { accepted: false, reason: `DOM dimensions too small (${Math.round(rect.width)}x${Math.round(rect.height)})` };
    }

    if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= window.innerHeight || rect.left >= window.innerWidth) {
      return { accepted: false, reason: "completely outside viewport" };
    }

    const compStyle = window.getComputedStyle(v);
    if (compStyle.display === "none" || compStyle.visibility === "hidden" || parseFloat(compStyle.opacity) === 0) {
      return { accepted: false, reason: "hidden by CSS display/visibility/opacity" };
    }

    if (vw < 20 || vh < 20) {
      if (v.srcObject && v.srcObject.getVideoTracks && v.srcObject.getVideoTracks().length > 0) {
        const track = v.srcObject.getVideoTracks()[0];
        if (!track.enabled || track.muted || track.readyState !== "live") {
          return { accepted: false, reason: "MediaStream track is muted/disabled" };
        }
      } else {
        return { accepted: false, reason: `zero video resolution (${vw}x${vh})` };
      }
    }

    return { accepted: true, reason: `valid (${vw}x${vh}, DOM ${Math.round(rect.width)}x${Math.round(rect.height)})` };
  }

  // ==========================================
  // Multi-Tile Lifecycle Management
  // ==========================================

  function createTileContext(video, index) {
    tileIdCounter++;
    const tileId = `tile_${tileIdCounter}`;
    const participantName = extractParticipantName(video, index);
    const isSelfView = participantName.includes("You");

    getOrCreateOverlayRoot();

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
      mirrorMode: "auto",
      currentFlipped: false,
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
        tile.lastMirrorTs = 0;
        const { shouldFlip } = resolveTileMirrorState(tile);
        tile.currentFlipped = shouldFlip;
        updateTileFlipButton(tile);
      });
    }

    if (window.ResizeObserver) {
      tile.resizeObserver = new ResizeObserver(() => {
        tile.lastMirrorTs = 0;
        const { shouldFlip } = resolveTileMirrorState(tile);
        tile.currentFlipped = shouldFlip;
        updateTileFlipButton(tile);
        repositionTileBadge(tile);
      });
      tile.resizeObserver.observe(video);
      if (video.parentElement) {
        tile.resizeObserver.observe(video.parentElement);
      }
    }

    tile.videoObserver = new MutationObserver(() => {
      if (writingOwnStyle || observerPaused) return;

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
      const { shouldFlip } = resolveTileMirrorState(tile);
      tile.currentFlipped = shouldFlip;
      updateTileFlipButton(tile);
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
    tile.trackedFaces.clear();
  }

  function scanAndSyncActiveTiles() {
    if (!config.enabled || document.hidden) return;

    const allVideos = Array.from(document.querySelectorAll("video"));
    const activeCandidates = allVideos.filter((v) => evaluateVideoCandidate(v).accepted);

    // Fallback: If no candidate passed strict check, pick largest visible video
    if (activeCandidates.length === 0 && allVideos.length > 0) {
      const visible = allVideos.filter((v) => {
        const r = v.getBoundingClientRect();
        return r.width >= 40 && r.height >= 40 && r.bottom > 0 && r.right > 0;
      });
      if (visible.length > 0) {
        visible.sort((a, b) => {
          const rA = a.getBoundingClientRect();
          const rB = b.getBoundingClientRect();
          return (rB.width * rB.height) - (rA.width * rA.height);
        });
        activeCandidates.push(visible[0]);
      }
    }

    // 1. Remove tiles whose videos no longer exist or are inactive
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
        const existingTile = tileContexts.get(v);
        const curName = extractParticipantName(v, idx + 1);
        if (existingTile.participantName !== curName) {
          existingTile.participantName = curName;
          const nameSpan = existingTile.hudBadge.querySelector(".hv-badge-participant-name");
          if (nameSpan) nameSpan.textContent = curName;
        }
      }
    });

    // 3. Update HUD badge positions & check mirror state (re-checked every 1s)
    const now = performance.now();
    for (const [, tile] of tileContexts) {
      repositionTileBadge(tile);
      if (now - tile.lastMirrorTs >= 1000) {
        const { shouldFlip } = resolveTileMirrorState(tile);
        if (tile.currentFlipped !== undefined && tile.currentFlipped !== shouldFlip) {
          tile.currentFlipped = shouldFlip;
          updateTileFlipButton(tile);
        } else {
          tile.currentFlipped = shouldFlip;
        }
      }
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
      tile.trackedFaces.clear();
    }
    if (hudCtx && hudCanvas) {
      hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
    }
  }

  // =========================================================================
  // Multi-Face IoU & Center-Distance Matching
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

  // =========================================================================
  // Tile Inference Processing & Per-Face Target Updating
  // =========================================================================

  function processTileDetections(tile, data, capture, clientLatency) {
    lastBackendResponse = data;
    const video = tile.video;
    const badge = tile.hudBadge;
    if (!video || !badge) return;

    const { verdict, face_detected, latency_ms, emotion } = data;
    const rect = video.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const fw = data.frame_width || capture.capturedW || 320;
    const fh = data.frame_height || capture.capturedH || 180;

    const { shouldFlip, modeLabel } = resolveTileMirrorState(tile);
    updateTileFlipButton(tile);

    const style = window.getComputedStyle(video);
    const objectFit = style.objectFit || "cover";

    // ZERO FACES DETECTED IN THIS TILE: Clear targets & set searching state
    if (!face_detected || verdict === "no_face" || !data.faces || data.faces.length === 0) {
      tile.consecutiveNoFace++;
      tile.lastVerdict = "no_face";

      badge.className = "hv-hud-badge hv-no-face";
      const titleSpan = badge.querySelector(".hv-badge-title");
      const scoreSpan = badge.querySelector(".hv-badge-score");
      if (titleSpan) titleSpan.textContent = "NO FACE";
      if (scoreSpan) scoreSpan.textContent = "Position face in camera";

      tile.trackedFaces.clear();
      repositionTileBadge(tile);
      return;
    }

    tile.consecutiveNoFace = 0;
    const rawFaces = data.faces;

    // Use unified shared box-mapping engine to map each detected face to screen
    const candidates = rawFaces.map((det) => {
      const mapped = boxEngine.mapFaceToScreen(
        det,
        fw,
        fh,
        rect,
        video.videoWidth || rect.width,
        video.videoHeight || rect.height,
        objectFit,
        shouldFlip
      );

      return {
        det,
        tgtX1: mapped.screenX,
        tgtY1: mapped.screenY,
        tgtX2: mapped.screenX + mapped.width,
        tgtY2: mapped.screenY + mapped.height,
        width: mapped.width,
        height: mapped.height,
        centerX: mapped.centerX,
        centerY: mapped.centerY,
        rawNormBox: mapped.normBox
      };
    });

    // Pairwise matching with existing tracked faces
    const existingFaces = Array.from(tile.trackedFaces.values());
    const matchedExistingIds = new Set();
    const matchedCandidateIndices = new Set();

    const matchPairs = [];
    existingFaces.forEach((f) => {
      candidates.forEach((c, cIdx) => {
        const iou = boxIoU(f.lastNormBox, c.rawNormBox);
        const dist = centerDistance(f.lastNormBox, c.rawNormBox);
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

    matchPairs.sort((a, b) => b.score - a.score);
    matchPairs.forEach((pair) => {
      if (!matchedExistingIds.has(pair.face.id) && !matchedCandidateIndices.has(pair.cIdx)) {
        matchedExistingIds.add(pair.face.id);
        matchedCandidateIndices.add(pair.cIdx);

        const c = candidates[pair.cIdx];
        updateTrackedFace(pair.face, c);
      }
    });

    candidates.forEach((c, cIdx) => {
      if (!matchedCandidateIndices.has(cIdx)) {
        const newFace = createTrackedFace(tile, c);
        updateTrackedFace(newFace, c);
      }
    });

    for (const [id, face] of tile.trackedFaces) {
      if (!matchedExistingIds.has(id)) {
        face.missedCount = (face.missedCount || 0) + 1;
        if (face.missedCount >= 2) {
          tile.trackedFaces.delete(id);
        }
      } else {
        face.missedCount = 0;
      }
    }

    repositionTileBadge(tile);

    // Update Tile Badge
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

  function createTrackedFace(tile, candidate) {
    const existingIndices = Array.from(tile.trackedFaces.values()).map((f) => f.index);
    let targetIndex = 1;
    while (existingIndices.includes(targetIndex)) targetIndex++;

    const faceId = `${tile.id}_face_${targetIndex}`;

    const face = {
      id: faceId,
      index: targetIndex,
      label: `Face #${targetIndex}`,
      curX1: candidate.tgtX1,
      curY1: candidate.tgtY1,
      curX2: candidate.tgtX2,
      curY2: candidate.tgtY2,
      tgtX1: candidate.tgtX1,
      tgtY1: candidate.tgtY1,
      tgtX2: candidate.tgtX2,
      tgtY2: candidate.tgtY2,
      opacity: 0.1,
      lastNormBox: candidate.rawNormBox,
      verdict: "authentic",
      confidence: 100.0,
      emotion: "Focused / Calm 👤",
      missedCount: 0
    };

    tile.trackedFaces.set(faceId, face);
    return face;
  }

  function updateTrackedFace(face, candidate) {
    face.lastNormBox = candidate.rawNormBox;
    face.emotion = candidate.det.emotion || "Focused / Calm 👤";
    face.verdict = candidate.det.verdict || "authentic";
    face.confidence = candidate.det.confidence || 100.0;
    face.tgtX1 = candidate.tgtX1;
    face.tgtY1 = candidate.tgtY1;
    face.tgtX2 = candidate.tgtX2;
    face.tgtY2 = candidate.tgtY2;

    // Center displacement check: Snap immediately on big movement (>15% width), smooth on jitter
    const curWidth = Math.max(1, face.curX2 - face.curX1);
    const curCx = (face.curX1 + face.curX2) / 2;
    const tgtCx = (face.tgtX1 + face.tgtX2) / 2;
    if (Math.abs(tgtCx - curCx) / curWidth > 0.15) {
      face.curX1 = face.tgtX1;
      face.curY1 = face.tgtY1;
      face.curX2 = face.tgtX2;
      face.curY2 = face.tgtY2;
    }
  }

  // =========================================================================
  // 60 FPS Decoupled Smooth AF-LOCK Reticle Renderer (Zero Layout Thrashing)
  // =========================================================================

  function render60FPSHUD() {
    if (!config.enabled) {
      if (hudCtx && hudCanvas) {
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
      }
      hudAnimId = requestAnimationFrame(render60FPSHUD);
      return;
    }

    if (hudCanvas && hudCtx) {
      resizeHUDCanvas();
      hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);

      const LERP = 0.42;

      for (const [, tile] of tileContexts) {
        if (!tile.video || !document.body.contains(tile.video)) continue;

        for (const [, f] of tile.trackedFaces) {
          // Smooth glide toward target box
          f.curX1 += (f.tgtX1 - f.curX1) * LERP;
          f.curY1 += (f.tgtY1 - f.curY1) * LERP;
          f.curX2 += (f.tgtX2 - f.curX2) * LERP;
          f.curY2 += (f.tgtY2 - f.curY2) * LERP;
          f.opacity = Math.min(1.0, f.opacity + 0.15);

          // Draw reference AF-LOCK corner brackets and crosshair using shared module
          boxEngine.drawAFLockReticle(
            hudCtx,
            f.curX1,
            f.curY1,
            f.curX2,
            f.curY2,
            f,
            f.index,
            f.opacity,
            hudCanvas.width
          );
        }
      }
    }

    hudAnimId = requestAnimationFrame(render60FPSHUD);
  }

  // =========================================================================
  // Priority & Round-Robin Tile Scheduler (Bounded Load, Zero IPC Flood)
  // =========================================================================

  function selectNextScheduledTile() {
    const now = performance.now();
    const candidates = [];

    for (const [, tile] of tileContexts) {
      if (tile.inFlight) continue;
      if (now - tile.lastScanTs < config.minTileScanIntervalMs) continue;

      const video = tile.video;
      if (!video || !document.body.contains(video) || video.paused || video.ended) continue;

      const rect = video.getBoundingClientRect();
      const area = rect.width * rect.height;
      if (area < 2000) continue;

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

    tile.inFlight = true;
    lastGlobalScanTs = now;
    const reqStart = performance.now();

    const watchdogTimer = setTimeout(() => {
      if (tile.inFlight) {
        tile.inFlight = false;
        tile.lastScanTs = performance.now();
      }
    }, 6000);

    try {
      const capture = captureCompactFrame(tile.video);
      if (!capture) {
        tile.inFlight = false;
        clearTimeout(watchdogTimer);
        return;
      }

      const { shouldFlip } = resolveTileMirrorState(tile);

      if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
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
              lastError = chrome.runtime.lastError ? chrome.runtime.lastError.message : (response ? response.error : "Unknown backend response");
              return;
            }

            const clientLatency = Math.round(performance.now() - reqStart);
            try {
              processTileDetections(tile, response.data, capture, clientLatency);
            } catch (err) {
              lastError = err.message;
              debugLog("Tile detection processing error:", err);
            }
          }
        );
      } else {
        clearTimeout(watchdogTimer);
        tile.inFlight = false;
      }
    } catch (err) {
      lastError = err.message;
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

  // Global DOM childList observer (Watches ONLY additions/removals of DOM nodes, NEVER attributes)
  globalChildObserver = new MutationObserver(() => {
    if (document.hidden || !config.enabled) return;
    scanAndSyncActiveTiles();
  });
  globalChildObserver.observe(document.body, { childList: true, subtree: true });

  // Programmatic telemetry and inspection helper
  window.__getDeepfakeShieldTrackedFaces = function () {
    const list = [];
    for (const [, tile] of tileContexts) {
      for (const [, trk] of tile.trackedFaces) {
        list.push({
          tileId: tile.id,
          participantName: tile.participantName,
          id: trk.id,
          targetIndex: trk.index || 1,
          curX1: trk.curX1,
          curY1: trk.curY1,
          curX2: trk.curX2,
          curY2: trk.curY2,
          centerX: (trk.curX1 + trk.curX2) / 2,
          centerY: (trk.curY1 + trk.curY2) / 2,
          width: trk.curX2 - trk.curX1,
          height: trk.curY2 - trk.curY1,
          verdict: trk.face ? trk.face.verdict : "uncertain",
          confidence: trk.face ? trk.face.confidence : 99,
          emotion: trk.face ? trk.face.emotion : "Focused / Calm 👤",
          opacity: trk.opacity
        });
      }
    }
    return list;
  };

  // Developer & Diagnostic Self-Test Console Command
  window.deepfakeShieldSelfTest = async function () {
    console.group("%c[Deepfake Shield Diagnostics & Self-Test]", "color:#10b981;font-weight:bold;font-size:14px;");

    // 1. Backend Connectivity
    let backendStatus = "UNKNOWN";
    try {
      const resp = await fetch(`${config.backendUrl}/health`);
      if (resp.ok) {
        const h = await resp.json();
        backendStatus = `ONLINE (${h.status}, video: ${h.video_status}, device: ${h.device})`;
      } else {
        backendStatus = `HTTP ${resp.status}`;
      }
    } catch (err) {
      backendStatus = `OFFLINE (${err.message})`;
    }
    console.log(`%c• Backend Server (${config.backendUrl}):%c ${backendStatus}`, "font-weight:bold;", "color:#0ea5e9;");

    // 2. Video Candidate Audit
    const allVideos = Array.from(document.querySelectorAll("video"));
    console.log(`%c• Total <video> tags in DOM:%c ${allVideos.length}`, "font-weight:bold;", "color:#0ea5e9;");
    const videoAudit = allVideos.map((v, i) => {
      const evalRes = evaluateVideoCandidate(v);
      const r = v.getBoundingClientRect();
      return {
        Index: i + 1,
        VideoTag: `${v.tagName.toLowerCase()}${v.id ? "#" + v.id : ""}`,
        Accepted: evalRes.accepted ? "✅ YES" : "❌ NO",
        Reason: evalRes.reason,
        Resolution: `${v.videoWidth}x${v.videoHeight}`,
        DOMRect: `${Math.round(r.left)},${Math.round(r.top)} (${Math.round(r.width)}x${Math.round(r.height)})`,
        Parent: v.parentElement ? `${v.parentElement.tagName.toLowerCase()}.${v.parentElement.className.split(" ")[0]}` : "none"
      };
    });
    console.table(videoAudit);

    // 3. Active Tiles
    console.log(`%c• Active Tracked Tiles:%c ${tileContexts.size}`, "font-weight:bold;", "color:#0ea5e9;");
    const tileAudit = Array.from(tileContexts.values()).map((t) => {
      const m = resolveTileMirrorState(t);
      return {
        TileID: t.id,
        Participant: t.participantName,
        IsSelfView: t.isSelfView ? "YES" : "NO",
        MirrorState: `${m.shouldFlip ? "MIRRORED" : "NORMAL"} (${m.modeLabel})`,
        FacesTracked: t.trackedFaces.size,
        LastVerdict: t.lastVerdict,
        InFlight: t.inFlight ? "YES" : "NO"
      };
    });
    console.table(tileAudit);

    // 4. Last Detection Response Shape
    console.log("%c• Last Backend Response Sample:%c", "font-weight:bold;", "color:#0ea5e9;");
    console.dir(lastBackendResponse || "None received yet");

    if (lastError) {
      console.warn(`Last encountered error: ${lastError}`);
    }

    console.groupEnd();
    return {
      backendStatus,
      totalVideos: allVideos.length,
      activeTiles: tileContexts.size,
      lastBackendResponse,
      lastError
    };
  };

  // Initialize
  getOrCreateOverlayRoot();
  startSchedulerLoop();
  createDockPill();
  render60FPSHUD();
})();
