"""Build DAIC-WoZ trimodal features in the dvlog/ folder format.

Per participant (196 zips, AVEC2017 splits train/dev/test -> train/valid/test):
  acoustic: COVAREP.csv (100 Hz, 74-dim) -> 1 s mean-pool -> (T, 74) float32
  visual:   CLNF_features.txt (30 Hz, header: frame,timestamp,confidence,success,
            x0..x67,y0..y67) -> success==1 only, 1 s mean-pool, gaps ffill/bfill
            -> (T, 136) float32
  text:     TRANSCRIPT.csv (tab-sep) participant utterances joined ->
            bert-base-uncased, padding="max_length", max_length=512,
            last_hidden_state -> (512, 768) float32  [same as dvlog_transcript]
  T = min(T_acoustic, T_visual); labels.csv: index,label,duration,gender,fold
  (gender: DAIC Gender column, 0 -> 'f', 1 -> 'm'; only used for gender-filtered runs)

Usage: python build_daic_features.py [--device cuda:0] [--limit N]
Skips participants whose three .npy files already exist (idempotent).
"""
import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DAIC_RAW = Path(os.environ.get("DAIC_RAW_ROOT", "./data/DAIC_WOZ_raw"))
OUT_DIR = Path(os.environ.get("DAIC_DATA_ROOT",
                              os.path.join(os.environ.get("DEPM3_DATA_ROOT", "./data"), "daic_woz")))


def load_splits():
    rows = {}  # pid -> (label_int, gender, fold)
    for csv, bin_col, fold in [
        ("train_split_Depression_AVEC2017.csv", "PHQ8_Binary", "train"),
        ("dev_split_Depression_AVEC2017.csv", "PHQ8_Binary", "valid"),
        ("full_test_split.csv", "PHQ_Binary", "test"),
    ]:
        df = pd.read_csv(DAIC_RAW / csv)
        for _, r in df.iterrows():
            pid = int(r["Participant_ID"])
            rows[pid] = (int(r[bin_col]), "f" if int(r["Gender"]) == 0 else "m", fold)
    return rows


def pool_seconds(arr: np.ndarray, hz: int) -> np.ndarray:
    n_sec = arr.shape[0] // hz
    return arr[: n_sec * hz].reshape(n_sec, hz, arr.shape[1]).mean(axis=1)


def build_acoustic(z: zipfile.ZipFile, pid: int) -> np.ndarray:
    with z.open(f"{pid}_COVAREP.csv") as f:
        df = pd.read_csv(io.BytesIO(f.read()), header=None, dtype=np.float32)
    a = np.nan_to_num(df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return pool_seconds(a, 100)


def build_visual(z: zipfile.ZipFile, pid: int) -> np.ndarray:
    with z.open(f"{pid}_CLNF_features.txt") as f:
        df = pd.read_csv(io.BytesIO(f.read()), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    coord_cols = [f"x{i}" for i in range(68)] + [f"y{i}" for i in range(68)]
    df = df[df["success"] == 1]
    if df.empty:
        return np.zeros((0, 136), dtype=np.float32)
    sec = df["timestamp"].astype(float).astype(int)
    g = df[coord_cols].astype(np.float32).groupby(sec).mean()
    full = g.reindex(range(int(sec.max()) + 1)).ffill().bfill()
    return full.to_numpy(dtype=np.float32)


def build_text(z: zipfile.ZipFile, pid: int, tok, bert, device: str) -> np.ndarray:
    import torch

    with z.open(f"{pid}_TRANSCRIPT.csv") as f:
        df = pd.read_csv(io.BytesIO(f.read()), sep="\t")
    utts = df[df["speaker"].str.strip().str.lower() == "participant"]["value"]
    text = " ".join(str(u).strip() for u in utts if isinstance(u, str) or not pd.isna(u))
    enc = tok(text, truncation=True, padding="max_length", max_length=512,
              return_tensors="pt").to(device)
    with torch.no_grad():
        h = bert(**enc).last_hidden_state[0].cpu().numpy().astype(np.float32)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N participants")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(args.device).eval()

    splits = load_splits()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_rows, skipped = [], []
    for n, (pid, (label, gender, fold)) in enumerate(sorted(splits.items())):
        if args.limit and n >= args.limit:
            break
        zpath = DAIC_RAW / f"{pid}_P.zip"
        if not zpath.exists():
            skipped.append((pid, "no zip"))
            continue
        pdir = OUT_DIR / str(pid)
        paths = {m: pdir / f"{pid}_{m}.npy" for m in ("acoustic", "visual", "text")}
        try:
            if all(p.exists() for p in paths.values()):
                T = min(np.load(paths["acoustic"], mmap_mode="r").shape[0],
                        np.load(paths["visual"], mmap_mode="r").shape[0])
            else:
                with zipfile.ZipFile(zpath) as z:
                    a = build_acoustic(z, pid)
                    v = build_visual(z, pid)
                    t = build_text(z, pid, tok, bert, args.device)
                T = min(a.shape[0], v.shape[0])
                if T == 0:
                    skipped.append((pid, f"empty modality a={a.shape} v={v.shape}"))
                    continue
                pdir.mkdir(exist_ok=True)
                np.save(paths["acoustic"], a[:T])
                np.save(paths["visual"], v[:T])
                np.save(paths["text"], t)
            label_rows.append((pid, "depression" if label else "normal", T, gender, fold))
            print(f"[{n + 1}/{len(splits)}] {pid} fold={fold} label={label} T={T}", flush=True)
        except Exception as e:  # noqa: BLE001 — keep going, report at end
            skipped.append((pid, repr(e)[:200]))
            print(f"[{n + 1}/{len(splits)}] {pid} SKIP: {e}", flush=True)

    with open(OUT_DIR / "labels.csv", "w") as f:
        f.write("index,label,duration,gender,fold\n")
        for pid, label, T, gender, fold in label_rows:
            f.write(f"{pid},{label},{T},{gender},{fold}\n")
    print(f"done: {len(label_rows)} participants, {len(skipped)} skipped")
    for pid, why in skipped:
        print(f"  skipped {pid}: {why}")


if __name__ == "__main__":
    main()
