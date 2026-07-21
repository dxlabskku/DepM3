# Evaluation


Training writes a run directory `<save_dir>/<dataset>/multienc_..._<timestamp>/`
containing `config.json` and `checkpoints/best_model.pt`. That directory is the
`--run_dir` for the evaluation scripts (run from `evaluation/`):

```bash
# A. global missing-modality conditions (7: full, w/o {a,v,t}, {a,v,t}-only)
# B. frame-level local masking (segment / random, ratios 0.1/0.3/0.5)
# C. Gaussian corruption (sigma 0.5/1.0)
python eval_missing.py --run_dir <run_dir> --data_dir $DEPM3_DATA_ROOT/dvlog --device cuda:0

# segment masking averaged over 5 mask placements
python eval_segment_multiplace.py --run_dir <run_dir> --out seg_multiplace.json

# efficiency (params / latency / peak memory / GFLOPs) of the paper config
# (Post-MoE 6/top-2 + Pre-FiLM 4/top-2 are hard-coded to the paper checkpoint config)
python measure_efficiency.py --device cuda:0 --seq_lens 500 1000 2000 4000
```
