# DepM³ — Post-level Training (D-Vlog / LMVD / DAIC-WoZ)

Entry point: `main_multiencoder_ensemble.py` (see `--help` for all options).

## Environment

- Python 3.10, single NVIDIA GPU (CUDA 12.8 toolchain)
- `torch 2.11.0+cu128`
- `mamba_ssm 2.3.1`
- `causal_conv1d 1.6.1`

Install the CUDA packages manually (matching wheels from the PyTorch,
`state-spaces/mamba`, and `Dao-AILab/causal-conv1d` release pages), then
`pip install -r requirements.txt` for the rest.

Notes:

- `post_level/models/mamba/selective_scan_interface.py` (and its `user_level`
  copy) contains a **compatibility shim for `causal_conv1d >= 1.6`**, whose CUDA
  kernel API changed to an in-place 8-argument form. No patching of the
  installed packages is needed; the shim adapts at import time.
- If your user site-packages contain conflicting builds, run everything with
  `PYTHONNOUSERSITE=1`.

## Data layout

No dataset is redistributed here. All four must be obtained from their original
providers. Feature roots are resolved through environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `DEPM3_DATA_ROOT` | common data root | `./data` |
| `LMVD_DATA_ROOT`  | LMVD feature root (overrides) | `$DEPM3_DATA_ROOT/lmvd` |
| `MUD3_DATA_ROOT`  | MUD3 feature root (overrides) | `$DEPM3_DATA_ROOT/MUD3` |
| `DAIC_RAW_ROOT`   | raw DAIC-WoZ zips (for the builder) | `./data/DAIC_WOZ_raw` |
| `DAIC_DATA_ROOT`  | built DAIC features (overrides) | `$DEPM3_DATA_ROOT/daic_woz` |

### D-Vlog

Request the features from the original D-Vlog authors. Expected layout:

```
$DEPM3_DATA_ROOT/dvlog/
  labels.csv                      # index,label,duration,gender,fold
  <id>/<id>_acoustic.npy          # (T, 25)
  <id>/<id>_visual.npy            # (T, 136)
  <id>/<id>_text.npy              # (512, 768)
```

### LMVD

Request from the LMVD authors. Expected layout:

```
$LMVD_DATA_ROOT/
  labels.csv
  audio/<id>.npy                  # (T, 128)
  visual/<id>_visual.npy          # (T, 136)
  text/<id>_text.npy              # (512, 768)
```

### MUD3

Request from the MUD3 authors. Expected layout:

```
$MUD3_DATA_ROOT/
  labels.csv
  dep_feat.pkl
  nondep_feat.pkl
```

### DAIC-WoZ

Sign the license agreement with USC ICT to obtain the raw distribution (the 189
participant session archives + AVEC2017 split CSVs). Then build features into
the shared D-Vlog folder layout:

```bash
export DAIC_RAW_ROOT=/path/to/DAIC_WOZ_raw          # folder with the <pid>_P.zip files + split CSVs
export DEPM3_DATA_ROOT=./data                        # output goes to $DEPM3_DATA_ROOT/daic_woz
python data_preparation/build_daic_features.py --device cuda:0
```

This produces `<pid>_{acoustic,visual,text}.npy` (COVAREP 74-d, CLNF 136-d,
BERT transcript embedding) plus `labels.csv` with the AVEC2017 folds.
Notes: all DAIC runs use `--pos_weight 2.57` (class imbalance), and
`main_multiencoder_ensemble.py` requires a one-time self-symlink
`ln -s . "$DEPM3_DATA_ROOT/daic_woz/dvlog"`.
