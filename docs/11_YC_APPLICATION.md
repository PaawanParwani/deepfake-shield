# YC Application Draft Notes

Context: YC's Fall 2026 Requests for Startups includes "Proving You're Human" (rebuilding the trust layer of the internet; verifying a real human on a call, message or transaction, ideally without sacrificing privacy). Our project is a direct, early step toward that. Check yc.com/apply for current deadlines and batch timing.

## One-liner
We detect cloned voices and deepfake faces in real time, and stress-test the detectors against attackers.

## Problem
Deepfake fraud is exploding and nobody can tell who is real on a call. A single fake video meeting cost a company $25M.

## Solution
A verification layer that scores audio and video for synthetic manipulation and is hardened against adversarial evasion.

## Why now
Voice clones and face swaps are cheap and realistic; every bank, enterprise and dating/marketplace app needs a trust signal.

## Traction (fill honestly)
- Working demo: yes/no
- Metrics: EER __, AUC __
- Users/pilots: none yet (say so; list who you will approach)

## Why us
Cybersecurity + data science founders; we built it as a research project, with a security-first evaluation most detectors lack.

## Market and wedge
Wedge: enterprise finance teams verifying high-value calls and transfers. Expand: banks, KYC/onboarding, marketplaces, social platforms.

## Competition (research and fill)
Existing deepfake-detection vendors. Our angle: adversarial robustness and transparency about failure modes.

## Honest risks to address
- Detector accuracy drops on unseen generators (generalization)
- Arms race with generative models
- Needs real-world data beyond academic benchmarks

## Application video (1 min)
Use the demo script in `docs/08_DEMO_SPEC.md`.
