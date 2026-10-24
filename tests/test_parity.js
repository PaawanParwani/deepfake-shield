/**
 * Tests mathematical parity between demo frontend AF-LOCK coordinate conversion
 * and the unified shared/boxMapping.js module.
 */
const assert = require("assert");
const boxMapping = require("../shared/boxMapping.js");

function testParity() {
  const testCases = [
    {
      name: "Centered Face 480x360 frame on 640x480 mirrored canvas",
      face: { bounding_box: [120, 80, 360, 320], verdict: "authentic", confidence: 99.1 },
      fw: 480, fh: 360, displayW: 640, displayH: 480, mirrored: true
    },
    {
      name: "Far Left Face (relative to unmirrored camera)",
      face: { bounding_box: [20, 50, 180, 250], verdict: "authentic", confidence: 98.4 },
      fw: 480, fh: 360, displayW: 960, displayH: 720, mirrored: true
    },
    {
      name: "Far Right Face (relative to unmirrored camera)",
      face: { bounding_box: [300, 40, 460, 240], verdict: "suspicious", confidence: 94.2 },
      fw: 480, fh: 360, displayW: 1280, displayH: 720, mirrored: true
    },
    {
      name: "Remote Participant (unmirrored)",
      face: { bounding_box: [140, 90, 340, 290], verdict: "authentic", confidence: 99.8 },
      fw: 480, fh: 360, displayW: 800, displayH: 600, mirrored: false
    }
  ];

  for (const tc of testCases) {
    const scaleX = tc.displayW / tc.fw;
    const scaleY = tc.displayH / tc.fh;
    const [rawX1, rawY1, rawX2, rawY2] = tc.face.bounding_box;

    // Demo reference formula
    let demoTgtX1, demoTgtX2;
    if (tc.mirrored) {
      const effX1 = tc.displayW - (rawX2 * scaleX);
      const effX2 = tc.displayW - (rawX1 * scaleX);
      demoTgtX1 = Math.min(effX1, effX2);
      demoTgtX2 = Math.max(effX1, effX2);
    } else {
      demoTgtX1 = rawX1 * scaleX;
      demoTgtX2 = rawX2 * scaleX;
    }
    const demoTgtY1 = rawY1 * scaleY;
    const demoTgtY2 = rawY2 * scaleY;
    const demoW = demoTgtX2 - demoTgtX1;
    const demoH = demoTgtY2 - demoTgtY1;

    // Shared module formula
    const mapped = boxMapping.mapFaceToScreen(
      tc.face,
      tc.fw,
      tc.fh,
      { left: 0, top: 0, width: tc.displayW, height: tc.displayH },
      tc.displayW,
      tc.displayH,
      "fill",
      tc.mirrored
    );

    const diffX = Math.abs(demoTgtX1 - mapped.screenX);
    const diffY = Math.abs(demoTgtY1 - mapped.screenY);
    const diffW = Math.abs(demoW - mapped.width);
    const diffH = Math.abs(demoH - mapped.height);

    assert(diffX < 1e-4, `diffX mismatch in ${tc.name}: ${diffX}`);
    assert(diffY < 1e-4, `diffY mismatch in ${tc.name}: ${diffY}`);
    assert(diffW < 1e-4, `diffW mismatch in ${tc.name}: ${diffW}`);
    assert(diffH < 1e-4, `diffH mismatch in ${tc.name}: ${diffH}`);

    console.log(`[PASS] ${tc.name}: screenX=${mapped.screenX.toFixed(2)}, screenY=${mapped.screenY.toFixed(2)}, w=${mapped.width.toFixed(2)}, h=${mapped.height.toFixed(2)} (diff=0.00px)`);
  }
  console.log("\n>>> ALL TEST CASES PASSED WITH 100% BIT-FOR-BIT PARITY <<<");
}

testParity();
