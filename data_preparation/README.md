# DAIC-WoZ Feature Builder

Sign the license agreement with USC ICT to obtain the raw distribution, then:

```bash
export DAIC_RAW_ROOT=/path/to/DAIC_WOZ_raw   # <pid>_P.zip files + AVEC2017 split CSVs
export DEPM3_DATA_ROOT=./data                # output: $DEPM3_DATA_ROOT/daic_woz
python build_daic_features.py --device cuda:0
```

Produces `<pid>_{acoustic,visual,text}.npy` (COVAREP 74-d, CLNF 136-d, BERT
transcript embedding) and `labels.csv` with the AVEC2017 folds. All DAIC runs
use `--pos_weight 2.57` (class imbalance). `main_multiencoder_ensemble.py`
requires a one-time self-symlink: `ln -s . "$DEPM3_DATA_ROOT/daic_woz/dvlog"`.
