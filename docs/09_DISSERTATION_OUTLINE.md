# Dissertation Outline

1. **Introduction**: problem, motivation (deepfake fraud), research questions, contributions
2. **Background and Related Work**: audio anti-spoofing (ASVspoof, RawNet2, AASIST, wav2vec2 methods), deepfake video detection (FaceForensics++, EfficientNet-based detectors), adversarial examples in audio and vision
3. **Threat Model**: attacker goals, capabilities (white-box, black-box), assumptions, out-of-scope threats
4. **Methodology**
   - 4.1 Audio detector
   - 4.2 Video detector
   - 4.3 Adversarial attack setup
   - 4.4 Combined decision logic
5. **Implementation**: stack, data pipelines, training setup, demo system
6. **Results**
   - 6.1 Clean performance (EER, AUC tables)
   - 6.2 Robustness under attack (degradation curves)
   - 6.3 Ablations (e.g. frozen vs fine-tuned backbone)
7. **Discussion**: what the robustness gap means, practical implications, limitations (dataset bias, generalization, no live-call test)
8. **Conclusion and Future Work**: defenses, learned fusion, live-call integration, multimodal consistency (lip-sync)
9. **References**
10. **Appendices**: configs, extra plots, ethics statement

## Research questions (draft)
- RQ1: How accurately can fine-tuned pretrained models detect synthetic voice and face video?
- RQ2: How much does detection performance degrade under adversarial perturbation for each modality?
- RQ3: Do the two modalities fail in similar or different ways, and does combining them help?
