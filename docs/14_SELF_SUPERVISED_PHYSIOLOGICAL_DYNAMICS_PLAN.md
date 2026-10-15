# Self-Supervised Physiological & Affective Dynamics Architecture (Self-Learning Deepfake Defense)

## Executive Summary

Current deepfake detectors rely on static spatial artifacts (blur, pixel blending seams) that quickly become obsolete as generative models (Diffusion, Sora, Flux, LivePortrait, V-Express) improve in resolution.

This architecture introduces **NeuroPulse & ChronoFace**: a **self-training, continually evolving biometric countermeasure** that detects deepfakes by monitoring **involuntary human physiological dynamics**—eye micro-movements, spontaneous blink kinematics, facial muscle action units (FACS), emotion-prosody synchronization, and remote photoplethysmography (rPPG subcutaneous cardiac pulse).

Crucially, the system features a **Self-Training Engine** where the model learns autonomously from observing real-world faces over time using **Temporal Contrastive Learning (DINO-Temporal)** and **Physiological Pseudo-Labeling with Experience Replay**.

---

## 1. The Core Scientific Principle: Involuntary Human Micro-Signals

Generative AI models synthesize frames independently or across short latent temporal interpolations. They fundamentally lack an underlying biological organism: they have no central nervous system, no heart pumping blood, and no involuntary oculomotor muscle contractions.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Human Video Stream (Webcam / Meet)                    │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────────┐           ┌──────────────────┐             ┌──────────────────┐
│  Ocular Dynamics │           │  Facial Action   │             │   Subcutaneous   │
│  & Blinking      │           │  Units (FACS)    │             │   Pulse (rPPG)   │
├──────────────────┤           ├──────────────────┤             ├──────────────────┤
│ - Eye Aspect     │           │ - 18 Micro-AUs   │             │ - Blood Volume   │
│   Ratio (EAR)    │           │ - Duchenne Smile │             │   Pulse (BVP)    │
│ - Corneal 3D     │           │   Consistency    │             │ - Forehead/Cheek │
│   Specular Glint │           │ - Affect-Prosody │             │   Peak Periodicity│
│ - Saccade Speed  │           │   Sync (Audio)   │             │   (0.75-2.5 Hz)  │
└────────┬─────────┘           └────────┬─────────┘             └────────┬─────────┘
         │                              │                                │
         └──────────────────────────────┼────────────────────────────────┘
                                        ▼
                   ┌────────────────────────────────────────┐
                   │    Biometric Temporal Transformer      │
                   │ (Cross-Attention Spatio-Temporal Model)│
                   └────────────────────┬───────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
               [Self-Training Gate]          [Live Real-Time Score]
             High Confidence Filtering      🟢 Verified Human (99.6%)
             (P_bio > 0.98 or P_fake > 0.95)🔴 Deepfake Alert (99.2%)
                         │
                         ▼
             [Experience Replay Buffer]
                         │
                         ▼
             [Continual Learning Engine]
             (EWC Self-Supervised Updates)
```

---

## 2. Physiological Biometric Channels

### A. Ocular Dynamics & Blinking Kinematics
1. **Eye Aspect Ratio (EAR):**
   $$\text{EAR}(t) = \frac{\|p_2(t) - p_6(t)\| + \|p_3(t) - p_5(t)\|}{2 \|p_1(t) - p_4(t)\|}$$
   where $p_1 \dots p_6$ are eye contour landmarks from MediaPipe FaceMesh.
2. **Biological Blink Profile:**
   - Normal human spontaneous blinking follows an asymmetric trajectory: rapid closing phase ($70-100\text{ ms}$) followed by a slower opening phase ($150-300\text{ ms}$).
   - Deepfakes produce either:
     - Non-closing eyes (blinking omission).
     - Abrupt step-function cuts (framerate desynchronization).
     - Symmetric linear blinks (mathematical interpolation rather than biological eyelid muscle contraction).
3. **Corneal Specular Highlight Symmetry:**
   - When light strikes human eyes, the corneal reflections (Purkinje images) in both pupils are geometrically locked to the 3D position of the room light source.
   - Generative models synthesize left and right eyes semi-independently, generating inconsistent specular reflection angles and highlight counts.

### B. Facial Action Unit (FACS) & Emotional Congruence
1. **Micro-Expression Action Units:**
   - Tracking 18 key Action Units: AU1 (Inner Brow Raiser), AU2 (Outer Brow Raiser), AU4 (Brow Lowerer), AU6 (Cheek Raiser), AU12 (Lip Corner Puller - Zygomaticus), AU25 (Lips Part).
2. **Duchenne vs. Non-Duchenne Smile Authenticity:**
   - Authentic emotional smiling activates **both AU12 (mouth) and AU6 (orbicularis oculi crinkling around the eyes)**.
   - Deepfakes routinely suffer from "affective dissociation": the mouth smiles while the periocular region remains completely rigid.
3. **Affect-Acoustic Congruency:**
   - Cross-referencing visual facial emotion (valence/arousal from expression) with acoustic prosody (pitch $F_0$, energy, and speech rate from `wav2vec2`).

### C. Remote Photoplethysmography (rPPG Cardiac Pulse)
1. **Biological Blood Volume Pulse Extraction:**
   - Ambient light penetrates the epidermis and is selectively absorbed by hemoglobin in capillary beds.
   - Blood volume fluctuations modulate skin pixel chrominance in the green/red spectrum synchronously with ventricular contraction:
   $$\text{CHROM} = 3 R - 2 G$$
2. **Spatial Pulse Coherence:**
   - In a living human, cardiac pulse waves travel synchronously across the forehead, left cheek, and right cheek with near-zero phase disparity.
   - Deepfakes disrupt spatial rPPG continuity: synthetic skin patches lack coherent heartbeat periodicities in the $0.75 - 2.5\text{ Hz}$ band ($45 - 150\text{ BPM}$).

---

## 3. The Self-Training Mechanism: How the Model Learns on Its Own

```
                    ┌────────────────────────────────────────┐
                    │       Unlabeled Video Stream Pool      │
                    │   (Live Calls, Web Video, User Feeds)  │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │      Biometric Multi-Channel Oracle    │
                    │  (Blink EAR + rPPG SNR + FACS Harmony) │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       [High Physiological Consistency]         [Physiological Anomaly]
          P_bonafide > 0.98                       P_synthetic > 0.95
          (Clean Blink + Coherent rPPG)          (Static Glint + Flatline rPPG)
                    │                                       │
                    ▼                                       ▼
       [Pseudo-Label: BONAFIDE]                [Pseudo-Label: DEEPFAKE]
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │     Curated Experience Replay Buffer   │
                    │    (Balanced Class Diversity Sampler)  │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       Continual Learning Engine        │
                    │    Elastic Weight Consolidation (EWC)  │
                    │   + Cosine Knowledge Distillation Loss │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       Updated Model Checkpoint         │
                    │   (Auto-backed up to Drive & HF Hub)   │
                    └────────────────────────────────────────┘
```

1. **Self-Supervised Bootstrap:**
   - The base model is initialized with the Dual-Stream CNN trained on FaceForensics++ C23 and Celeb-DF v2.
2. **Autonomous Observation & Pseudo-Labeling:**
   - As the system observes faces (in meeting calls or unannotated video archives), it computes the **Physiological Consistency Score** ($S_{\text{physio}}$) by fusing EAR blink kinematics, spatial rPPG SNR, and AU coherence.
   - **Strict Filtering Gate:**
     - If $S_{\text{physio}} \ge 0.98$ and spatial quality is high $\rightarrow$ Pseudo-labeled as **Bonafide**.
     - If spatial realism is high but $S_{\text{physio}} \le 0.15$ (e.g. flatline cardiac pulse or frozen periocular eyes) $\rightarrow$ Pseudo-labeled as **Deepfake**.
     - Ambiguous cases ($0.15 < S_{\text{physio}} < 0.98$) are discarded to maintain 100% clean self-training data.
3. **Experience Replay Buffer:**
   - High-confidence pseudo-labeled sequences are placed into an FIFO Experience Replay Buffer containing diverse skin tones, room lighting, and camera angles.
4. **Continual Adaptation with EWC (Elastic Weight Consolidation):**
   - Standard fine-tuning causes *catastrophic forgetting* (the model forgets previously learned deepfake artifacts).
   - EWC adds a quadratic penalty proportional to the Fisher Information Matrix ($F_i$) of previous weights:
   $$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{current}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{i,\text{old}})^2$$
   - This protects foundational biometric weights while allowing new neurons to adapt to novel lighting and compression artifacts autonomously.

---

## 4. Integration into Real-Time Extension, SaaS Web App & Mobile

1. **Client-Side MediaPipe FaceMesh (60 FPS):**
   - Calculates 468 3D landmarks in real-time in JavaScript (WebAssembly) or iOS Swift (Vision).
   - Extracts EAR and AU features directly inside the client without transmitting raw video to cloud servers.
2. **Cloud Microservice & Self-Training Worker:**
   - Lightweight FastAPI backend running on Hugging Face Spaces / Render.
   - Background worker processes accumulated Experience Replay clips during idle periods (e.g., nightly batch update) and pushes updated model weights to Hugging Face Model Hub.
3. **Native iOS & Android Deployment:**
   - **iOS:** CoreML model executes on the Apple Neural Engine (ANE) with on-device CoreML continual training APIs (`MLUpdateTask`).
   - **Android:** ONNX Runtime Mobile with NNAPI acceleration.

---

## 5. Research Paper Publication Strategy

### Title
*NeuroPulse: Self-Supervised Biometric Deepfake Detection via Continuous Involuntary Micro-Physiological Dynamics and Test-Time Adaptation*

### Target Conferences / Journals
- **IEEE Transactions on Information Forensics and Security (T-IFS)** (IF: 7.2)
- **ACM Multimedia (ACM MM)** (Oral Track)
- **IEEE/CVF CVPR Workshop on Media Forensics (WMF)**
- **Interspeech** (Audiovisual Anti-Spoofing Track)

### Novel Technical Contributions
1. First framework to unify ocular kinematics, spatial rPPG cardiac coherence, and FACS facial muscle harmony into a single temporal attention transformer.
2. An autonomous self-training loop that harvests real-world video streams to adapt to new camera sensors and compression standards without manual annotations.
3. Demonstration that biological micro-signals remain robust even when spatial GAN/Diffusion textures completely fool spatial-only detectors.
