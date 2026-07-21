# DepM³

**DepM³: Modality-Agnostic Mixture-of-Experts with Mamba for Depression Detection**

---

## Abstract

Multimodal depression detection has gained significant attention, yet most existing approaches assume complete modality availability at inference time. Addressing this limitation requires both adaptive fusion that adjusts computation based on which modalities are available and temporal modeling for the long behavioral sequences typical of depression analysis, two capabilities that existing methods rarely provide jointly. We propose DepM$^3$, a modality-agnostic framework that unifies long-range temporal modeling with modality-aware conditional computation for robust multimodal depression detection. DepM$^3$ introduces a two-stage Mixture-of-Experts mechanism: (1) a Modality-aware MoE that adapts modality-specific features prior to fusion and (2) Post-CoSSM MoE that routes fused representations based on explicit modality presence scores enabling adaptive computation where experts specialize for different modality subsets. A diversity-regularized multi-encoder ensemble further captures complementary depression indicators, improving robustness under incomplete modality scenarios. Built on a Mamba-based state space model, the framework achieves linear-time sequence processing suited to the temporally diffuse patterns characteristic of depression. We conduct comprehensive experiments on two post-level benchmarks (D-Vlog and LMVD), one user-level longitudinal benchmark (MUD3), and one clinical interview benchmark (DAIC-WoZ), comparing against both full-modality and missing-modality-oriented baselines. DepM$^3$ achieves F1-scores of 84.82\%, 81.97\%, and 56.52\% on D-Vlog, MUD3, and DAIC-WoZ, respectively, outperforming full- and missing-modality baselines on these three benchmarks and remaining competitive on LMVD, and degrades gracefully under global, segment-level, and noise-corrupted modality conditions.

---

## Framework

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

DepM³ consists of three main components:

1. **Modality-aware MoE (Pre-CoSSM)**
   FiLM-conditioned expert routing that adapts each modality's features before fusion, informed by modality identity and missing flags.

2. **Presence-conditioned MoE (Post-CoSSM)**
   Expert routing over the fused representation, conditioned on continuous modality presence scores within the Mamba-based sequence encoder.

3. **Diversity-regularized Multi-encoder Ensemble**
   Independent encoders trained with a diversity objective and combined by attention-based late fusion for robustness under uncertain missing patterns.

---

## Repository Structure

```text
depm3_release/
├── post_level/          # DepM³ training: D-Vlog / LMVD / DAIC-WoZ
│   ├── README.md
│   ├── main_multiencoder_ensemble.py
│   ├── models/
│   ├── datasets/
│   └── config/
│
├── user_level/          # DepM³ training: MUD3 (user-level)
│   ├── README.md
│   └── main_mud3.py
│
├── evaluation/          # missing/corrupted-modality evaluation + efficiency
│   └── README.md
│
├── data_preparation/    # DAIC-WoZ feature builder
│   └── README.md
│
├── assets/
│   └── framework.png
│
└── README.md
```
