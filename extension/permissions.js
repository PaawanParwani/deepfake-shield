// Handler for Camera & Microphone Permission Onboarding
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("btnRequest");
  const statusBox = document.getElementById("statusBox");

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Requesting Access in Browser...";

    try {
      // Trigger native Chrome permission modal
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });

      // Stop stream immediately after permission is confirmed
      stream.getTracks().forEach((track) => track.stop());

      // Save granted state to extension storage
      if (chrome && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ permissionsGranted: true });
      }

      statusBox.className = "status-box success";
      statusBox.innerHTML = "<strong>✅ Camera & Microphone Authorized!</strong><br>The Deepfake Shield is now fully active. You can close this tab and return to Google Meet.";
      btn.style.display = "none";

      setTimeout(() => {
        window.close();
      }, 3000);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Retry Permission Request";
      statusBox.className = "status-box error";
      statusBox.innerHTML = `<strong>Permission Denied or Blocked:</strong><br>${err.message}<br><br>Please check your browser URL bar icon (camera icon on top right) and set to 'Always Allow'.`;
    }
  });
});
