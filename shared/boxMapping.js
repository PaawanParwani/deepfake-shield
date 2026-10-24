/**
 * Deepfake Shield — Shared Box Mapping & Geometry Module
 * Used by BOTH the web demo (demo/frontend/index.html) and Google Meet extension.
 *
 * Implements the exact AF-LOCK coordinate transformation between backend frame space
 * and on-screen display coordinates, with full support for object-fit (cover, contain, fill)
 * and horizontal CSS mirroring.
 */
(function (global) {
  "use strict";

  /**
   * Computes rendered video geometry within its container, compensating for CSS object-fit.
   *
   * @param {number} elemW - Display width of video element.
   * @param {number} elemH - Display height of video element.
   * @param {number} videoW - Native video stream width.
   * @param {number} videoH - Native video stream height.
   * @param {string} objectFit - CSS object-fit ('cover', 'contain', or 'fill').
   * @returns {{ renderW: number, renderH: number, offsetX: number, offsetY: number }}
   */
  function computeRenderGeometry(elemW, elemH, videoW, videoH, objectFit = "cover") {
    const vw = videoW || elemW || 640;
    const vh = videoH || elemH || 480;

    if (elemW <= 0 || elemH <= 0 || vw <= 0 || vh <= 0) {
      return { renderW: elemW, renderH: elemH, offsetX: 0, offsetY: 0 };
    }

    const videoRatio = vw / vh;
    const elemRatio = elemW / elemH;

    let renderW = elemW;
    let renderH = elemH;
    let offsetX = 0;
    let offsetY = 0;

    if (objectFit === "contain") {
      if (elemRatio > videoRatio) {
        // Pillarboxed: letterbox on left and right
        renderH = elemH;
        renderW = elemH * videoRatio;
        offsetX = (elemW - renderW) / 2;
        offsetY = 0;
      } else {
        // Letterboxed: letterbox on top and bottom
        renderW = elemW;
        renderH = elemW / videoRatio;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      }
    } else if (objectFit === "cover") {
      if (elemRatio > videoRatio) {
        // Cropped top and bottom
        renderW = elemW;
        renderH = elemW / videoRatio;
        offsetX = 0;
        offsetY = (elemH - renderH) / 2;
      } else {
        // Cropped left and right
        renderH = elemH;
        renderW = elemH * videoRatio;
        offsetX = (elemW - renderW) / 2;
        offsetY = 0;
      }
    }

    return { renderW, renderH, offsetX, offsetY };
  }

  /**
   * Converts a backend face detection into precise display coordinates.
   *
   * Standardizes on the backend's `bounding_box: [x1, y1, x2, y2]` in sent frame pixels,
   * with automatic fallback to normalized `norm_xywh` or `norm_bbox`.
   *
   * @param {Object} face - Face detection object from /predict/frame.
   * @param {number} frameW - Width of image frame sent to backend (e.g. 480 or 320).
   * @param {number} frameH - Height of image frame sent to backend (e.g. 360 or 180).
   * @param {Object} videoRect - Bounding client rect of video element { left, top, width, height }.
   * @param {number} videoW - Video stream native width.
   * @param {number} videoH - Video stream native height.
   * @param {string} objectFit - CSS object-fit ('cover', 'contain', 'fill').
   * @param {boolean} isMirrored - Whether the displayed video is CSS mirrored (e.g. scaleX(-1)).
   * @returns {{
   *   screenX: number,
   *   screenY: number,
   *   width: number,
   *   height: number,
   *   centerX: number,
   *   centerY: number,
   *   localX: number,
   *   localY: number,
   *   normBox: [number, number, number, number]
   * }}
   */
  function mapFaceToScreen(face, frameW, frameH, videoRect, videoW, videoH, objectFit = "cover", isMirrored = false) {
    const fw = frameW || 480;
    const fh = frameH || 360;
    const elemW = videoRect.width || fw;
    const elemH = videoRect.height || fh;

    // 1. Extract normalized bounding box [normX, normY, normW, normH]
    // The demo reference standardizes strictly on bounding_box: [x1, y1, x2, y2]
    let normX, normY, normW, normH;

    if (face.bounding_box && face.bounding_box.length >= 4) {
      const [bx1, by1, bx2, by2] = face.bounding_box;
      normX = bx1 / fw;
      normY = by1 / fh;
      normW = (bx2 - bx1) / fw;
      normH = (by2 - by1) / fh;
    } else if (face.norm_xywh && face.norm_xywh.length >= 4) {
      [normX, normY, normW, normH] = face.norm_xywh;
    } else if (face.norm_bbox && face.norm_bbox.length >= 4) {
      normX = face.norm_bbox[0];
      normY = face.norm_bbox[1];
      normW = face.norm_bbox[2] - face.norm_bbox[0];
      normH = face.norm_bbox[3] - face.norm_bbox[1];
    } else {
      normX = 0; normY = 0; normW = 0.5; normH = 0.5;
    }

    // Clamp normalized values within bounds [0, 1]
    normX = Math.max(0.0, Math.min(1.0, normX));
    normY = Math.max(0.0, Math.min(1.0, normY));
    normW = Math.max(0.01, Math.min(1.0 - normX, normW));
    normH = Math.max(0.01, Math.min(1.0 - normY, normH));

    // 2. Compute object-fit rendered dimensions & letterbox/crop offsets
    const geom = computeRenderGeometry(elemW, elemH, videoW, videoH, objectFit);

    // 3. Mirror compensation: If mirrored, flip horizontal position across rendered video width
    // x_screen_norm = 1.0 - (normX + normW)
    const effNormX = isMirrored ? (1.0 - (normX + normW)) : normX;

    // 4. Calculate local coordinates within video element
    const localX = geom.offsetX + (effNormX * geom.renderW);
    const localY = geom.offsetY + (normY * geom.renderH);
    const boxW = normW * geom.renderW;
    const boxH = normH * geom.renderH;

    // 5. Calculate absolute screen coordinates
    const screenX = (videoRect.left || 0) + localX;
    const screenY = (videoRect.top || 0) + localY;
    const centerX = screenX + boxW / 2;
    const centerY = screenY + boxH / 2;

    return {
      screenX,
      screenY,
      width: boxW,
      height: boxH,
      centerX,
      centerY,
      localX,
      localY,
      renderW: geom.renderW,
      renderH: geom.renderH,
      offsetX: geom.offsetX,
      offsetY: geom.offsetY,
      normBox: [normX, normY, normW, normH]
    };
  }

  /**
   * Draws the exact reference AF-LOCK corner brackets and crosshair on a 2D canvas context.
   *
   * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
   * @param {number} bx1 - Left screen coordinate.
   * @param {number} by1 - Top screen coordinate.
   * @param {number} bx2 - Right screen coordinate.
   * @param {number} by2 - Bottom screen coordinate.
   * @param {Object} face - Face detection with verdict, confidence, emotion.
   * @param {number} targetIndex - 1-based index (e.g. 1 for Target #1).
   * @param {number} opacity - Render opacity [0.0 - 1.0].
   * @param {number} canvasW - Canvas boundary width.
   */
  function drawAFLockReticle(ctx, bx1, by1, bx2, by2, face, targetIndex = 1, opacity = 1.0, canvasW = window.innerWidth) {
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

    // Corner brackets
    ctx.beginPath();
    ctx.moveTo(bx1, by1 + cornerLen); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + cornerLen, by1);
    ctx.moveTo(bx2 - cornerLen, by1); ctx.lineTo(bx2, by1); ctx.lineTo(bx2, by1 + cornerLen);
    ctx.moveTo(bx1, by2 - cornerLen); ctx.lineTo(bx1, by2); ctx.lineTo(bx1 + cornerLen, by2);
    ctx.moveTo(bx2 - cornerLen, by2); ctx.lineTo(bx2, by2); ctx.lineTo(bx2, by2 - cornerLen);
    ctx.stroke();

    // Center Crosshair
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();

    // Lock Badge
    const badgeText = isFake
      ? `🔴 AF-ALERT #${targetIndex}: DEEPFAKE ${face.confidence || 90}%`
      : `🟢 AF-LOCK #${targetIndex}: AUTHENTIC ${face.confidence || 99}%`;
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

  // Universal export
  const exports = {
    computeRenderGeometry,
    mapFaceToScreen,
    drawAFLockReticle
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = exports;
  }
  if (typeof window !== "undefined") {
    window.DeepfakeShieldBoxMapping = exports;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
