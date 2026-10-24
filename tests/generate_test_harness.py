import base64
from pathlib import Path

f1_path = Path("demo/backend/test_crop_raw.jpg")
f2_path = Path("demo/backend/test_crop_887.jpg")

b64_f1 = base64.b64encode(f1_path.read_bytes()).decode("ascii")
b64_f2 = base64.b64encode(f2_path.read_bytes()).decode("ascii")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Google Meet Multi-Tile Simulation Test</title>
  <link rel="stylesheet" href="../extension/content.css">
  <style>
    body {{
      margin: 0;
      padding: 20px;
      background: #202124;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    h1 {{
      font-size: 20px;
      margin-bottom: 8px;
    }}
    .subtitle {{
      color: #9aa0a6;
      font-size: 13px;
      margin-bottom: 20px;
    }}
    #meet-stage {{
      width: 100%;
      max-width: 1200px;
      margin: 0 auto;
    }}
    #meet-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      width: 100%;
      height: 380px;
      transition: all 0.3s ease;
    }}
    /* Layout modes */
    #meet-grid.spotlight-layout {{
      grid-template-columns: 2.5fr 1fr;
      grid-template-rows: 1fr 1fr;
    }}
    #meet-grid.spotlight-layout .meet-tile:first-child {{
      grid-row: span 2;
    }}
    .meet-tile {{
      position: relative;
      background: #3c4043;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .meet-tile video {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    /* Meet CSS Mirroring on Self-View */
    .meet-tile.mirrored video {{
      transform: matrix(-1, 0, 0, 1, 0, 0);
    }}
    .name-label {{
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: rgba(0, 0, 0, 0.65);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 500;
      z-index: 10;
      pointer-events: none;
    }}
    .controls {{
      max-width: 1200px;
      margin: 20px auto;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    button {{
      background: #1a73e8;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 4px;
      font-size: 13px;
      cursor: pointer;
      font-weight: 500;
    }}
    button:hover {{ background: #1557b0; }}
    #test-log {{
      max-width: 1200px;
      margin: 16px auto;
      background: #111;
      padding: 16px;
      border-radius: 8px;
      font-family: Consolas, monospace;
      font-size: 12px;
      line-height: 1.6;
      border: 1px solid #333;
      min-height: 120px;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <div id="meet-stage">
    <h1>Google Meet Multi-Tile &amp; Multi-Face Validation Harness</h1>
    <div class="subtitle">Simulating 3 tiles: Tile 1 (You - CSS Mirrored, 1 face), Tile 2 (Alice - Normal, 1 face), Tile 3 (Room - Normal, 2 faces)</div>

    <div class="controls">
      <button id="btn-tiled">Layout: Tiled (3x1)</button>
      <button id="btn-spotlight">Layout: Spotlight (1 large + 2 side)</button>
      <button id="btn-toggle-tile">Toggle Participant 2 (Leave/Join)</button>
      <button id="btn-resize">Trigger Window Resize</button>
      <button id="btn-report" style="background:#0f9d58">Capture System Telemetry &amp; Report</button>
    </div>

    <div id="meet-grid">
      <!-- Tile 1: Self-View (Mirrored, 1 Face) -->
      <div class="meet-tile mirrored" data-self-name="You" id="tile-self">
        <video id="v-self" autoplay playsinline muted></video>
        <div class="name-label XE8d1b">You</div>
      </div>

      <!-- Tile 2: Remote Participant (Normal, 1 Face) -->
      <div class="meet-tile" data-participant-id="user_alice" id="tile-alice">
        <video id="v-alice" autoplay playsinline muted></video>
        <div class="name-label XE8d1b">Alice Smith</div>
      </div>

      <!-- Tile 3: Conference Room (Normal, 2 Faces) -->
      <div class="meet-tile" data-participant-id="user_room" id="tile-room">
        <video id="v-room" autoplay playsinline muted></video>
        <div class="name-label XE8d1b">Conference Room 402 (2 Faces)</div>
      </div>
    </div>

    <div id="test-log">Initializing synthetic video streams and Deepfake Shield...</div>
  </div>

  <script>
    // In-browser mock bridge for chrome.runtime.sendMessage -> local backend /predict/frame
    if (typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.sendMessage) {{
      window.chrome = window.chrome || {{}};
      window.chrome.runtime = window.chrome.runtime || {{}};
      window.chrome.runtime.sendMessage = function(msg, callback) {{
        if (msg.type === "PREDICT_FRAME") {{
          fetch(msg.payload.backendUrl + "/predict/frame", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(msg.payload)
          }})
          .then(r => r.json())
          .then(data => callback && callback({{ success: true, data }}))
          .catch(err => callback && callback({{ success: false, error: err.message }}));
        }} else {{
          if (callback) callback({{ success: true }});
        }}
      }};
      window.chrome.storage = {{
        local: {{
          get: (keys, cb) => cb && cb({{ enabled: true, backendUrl: "http://127.0.0.1:8000" }}),
          set: (data, cb) => cb && cb()
        }},
        onChanged: {{ addListener: () => {{}} }}
      }};
    }}

    // Real Face Images loaded into Image objects
    const imgFace1 = new Image();
    imgFace1.src = "data:image/jpeg;base64,{b64_f1}";

    const imgFace2 = new Image();
    imgFace2.src = "data:image/jpeg;base64,{b64_f2}";

    function setupStream(videoEl, faceConfig) {{
      const c = document.createElement("canvas");
      c.width = 640;
      c.height = 360;
      const ctx = c.getContext("2d");

      function draw() {{
        ctx.fillStyle = "#1e293b";
        ctx.fillRect(0, 0, c.width, c.height);

        // Subtle gradient background
        const grad = ctx.createLinearGradient(0, 0, c.width, c.height);
        grad.addColorStop(0, "#0f172a");
        grad.addColorStop(1, "#1e293b");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, c.width, c.height);

        if (faceConfig.mode === "single") {{
          const img = faceConfig.img;
          if (img.complete && img.naturalWidth > 0) {{
            const fw = 180;
            const fh = Math.round(fw * (img.naturalHeight / img.naturalWidth));
            const fx = (c.width - fw) / 2;
            const fy = (c.height - fh) / 2;
            ctx.drawImage(img, fx, fy, fw, fh);
          }}
        }} else if (faceConfig.mode === "double") {{
          // Two faces side by side
          if (imgFace1.complete && imgFace1.naturalWidth > 0) {{
            const fw = 150;
            const fh = Math.round(fw * (imgFace1.naturalHeight / imgFace1.naturalWidth));
            ctx.drawImage(imgFace1, 100, (c.height - fh) / 2, fw, fh);
          }}
          if (imgFace2.complete && imgFace2.naturalWidth > 0) {{
            const fw = 150;
            const fh = Math.round(fw * (imgFace2.naturalHeight / imgFace2.naturalWidth));
            ctx.drawImage(imgFace2, 390, (c.height - fh) / 2, fw, fh);
          }}
        }}

        requestAnimationFrame(draw);
      }}

      draw();
      const stream = c.captureStream(30);
      videoEl.srcObject = stream;
      videoEl.play().catch(() => {{}});
    }}

    setupStream(document.getElementById("v-self"), {{ mode: "single", img: imgFace1 }});
    setupStream(document.getElementById("v-alice"), {{ mode: "single", img: imgFace2 }});
    setupStream(document.getElementById("v-room"), {{ mode: "double" }});

    // Layout buttons
    document.getElementById("btn-tiled").addEventListener("click", () => {{
      const grid = document.getElementById("meet-grid");
      grid.className = "";
      log("Switched to Tiled Layout (3x1)");
    }});

    document.getElementById("btn-spotlight").addEventListener("click", () => {{
      const grid = document.getElementById("meet-grid");
      grid.className = "spotlight-layout";
      log("Switched to Spotlight Layout (Tile 1 large, Tiles 2 & 3 sidebar)");
    }});

    let aliceRemoved = false;
    let aliceBackup = null;
    document.getElementById("btn-toggle-tile").addEventListener("click", () => {{
      const grid = document.getElementById("meet-grid");
      const alice = document.getElementById("tile-alice");
      if (!aliceRemoved && alice) {{
        aliceBackup = alice;
        grid.removeChild(alice);
        aliceRemoved = true;
        log("Participant 2 (Alice Smith) left meeting / camera off");
      }} else if (aliceBackup) {{
        grid.insertBefore(aliceBackup, document.getElementById("tile-room"));
        aliceRemoved = false;
        setupStream(document.getElementById("v-alice"), {{ mode: "single", img: imgFace2 }});
        log("Participant 2 (Alice Smith) re-joined meeting / camera on");
      }}
    }});

    document.getElementById("btn-resize").addEventListener("click", () => {{
      window.dispatchEvent(new Event("resize"));
      log("Dispatched window resize event");
    }});

    function log(msg) {{
      const l = document.getElementById("test-log");
      l.textContent += "\\n[" + new Date().toISOString().substring(11, 19) + "] " + msg;
      l.scrollTop = l.scrollHeight;
    }}
  </script>

  <!-- Injected Content Script -->
  <script src="../extension/content.js"></script>

  <script>
    // System Telemetry Reporter
    window.__collectDeepfakeShieldTelemetry = function() {{
      const tiles = Array.from(document.querySelectorAll(".hv-hud-badge"));
      const faceOverlays = Array.from(document.querySelectorAll(".hv-face-overlay"));
      const videos = Array.from(document.querySelectorAll("video"));

      return {{
        tileBadgesFound: tiles.length,
        badges: tiles.map(b => ({{
          id: b.id,
          title: b.querySelector(".hv-badge-title") ? b.querySelector(".hv-badge-title").textContent : "",
          name: b.querySelector(".hv-badge-participant-name") ? b.querySelector(".hv-badge-participant-name").textContent : "",
          score: b.querySelector(".hv-badge-score") ? b.querySelector(".hv-badge-score").textContent : "",
          flipLabel: b.querySelector(".hv-badge-flip-btn") ? b.querySelector(".hv-badge-flip-btn").textContent : "",
          display: window.getComputedStyle(b).display,
          rect: b.getBoundingClientRect()
        }})),
        facesCount: faceOverlays.length,
        faceOverlays: faceOverlays.map(f => ({{
          label: f.querySelector(".hv-tag-identity") ? f.querySelector(".hv-tag-identity").textContent : "",
          verdict: f.querySelector(".hv-tag-verdict") ? f.querySelector(".hv-tag-verdict").textContent : "",
          emotion: f.querySelector(".hv-tag-emotion") ? f.querySelector(".hv-tag-emotion").textContent : "",
          rect: f.getBoundingClientRect()
        }})),
        videoCount: videos.length
      }};
    }};

    document.getElementById("btn-report").addEventListener("click", () => {{
      const t = window.__collectDeepfakeShieldTelemetry();
      log("TELEMETRY SNAPSHOT:\\n" + JSON.stringify(t, null, 2));
    }});
  </script>
</body>
</html>
"""

Path("tests/test_meet_multi_tile.html").write_text(html_content, encoding="utf-8")
print("tests/test_meet_multi_tile.html generated successfully.")
