# Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Not enough time for both modalities | High | High | Compressed plan; drop to one modality done well |
| FF++ access approval delayed | Medium | High | Apply immediately; fall back to a smaller public deepfake set |
| Colab GPU limits interrupt training | Medium | Medium | Cache features, checkpoint every epoch, train on subsets, ask university for HPC |
| Poor baseline accuracy | Medium | Medium | Report honestly; low clean accuracy is still a valid dissertation result |
| Attack adaptation to wav2vec2 input is fiddly | Medium | Medium | Start with noise/compression attacks, then PGD |
| Integration of two systems fails | Medium | Medium | Ship two separate verdicts; fusion is stretch |
| Data leakage (frames of same video in train and test) | Medium | Very high | Split by video; enforce in `dataset.py` |
| Partner delay blocks the other | Low | High | Independent tracks, independently submittable |
| Overclaiming results in paper/YC | Medium | High | Placeholders until real numbers; state limitations |

## Dependencies to start immediately
1. FF++ access request
2. Colab Pro / university compute
3. Dataset downloads (large)
