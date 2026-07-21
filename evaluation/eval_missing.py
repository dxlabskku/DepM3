"""Missing / corrupted modality evaluation for trained DepM3-family checkpoints.

Evaluates a trained run (config.json + checkpoints/best_model.pt) under:
  A. Global conditions (paper Fig.2): full / w/o audio / w/o visual / w/o text /
     audio-only / visual-only / text-only          -> zero-fill (triggers the
     model's L1-based missing detection & learnable embeddings)
  B. Frame-level local masking (reviewer zKDH): zero a fraction of frames of one
     modality, contiguous segment or random frames, ratios 0.1/0.3/0.5
  C. Gaussian corruption (reviewer mbvp): additive noise on one modality
     (non-zero corruption that the L1 detector cannot flag), sigma 0.5/1.0

Works for any config trained by main_multiencoder_ensemble.py (full DepM3,
ensemble-only, single-encoder no-MoE) -> enables baseline comparison under
identical conditions.

Usage:
  python eval_missing.py --run_dir <results/.../multienc_...timestamp> \
      --data_dir $DEPM3_DATA_ROOT/dvlog --device cuda:0
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "post_level"))

import numpy as np
import torch

from datasets import get_dvlog_trimodal_dataloader, get_lmvd_trimodal_dataloader
from models.DepMamba_multiencoder_ensemble import DepMambaMultiEncoderEnsemble

VIS, AUD_DV, AUD_LMVD, TXT = 136, 25, 128, 768


def modality_slices(dataset):
    aud = AUD_LMVD if dataset == "lmvd" else AUD_DV
    return {  # concat order in loaders: [visual | audio | text]
        "visual": slice(0, VIS),
        "audio": slice(VIS, VIS + aud),
        "text": slice(VIS + aud, VIS + aud + TXT),
    }


def build_model(cfg):
    return DepMambaMultiEncoderEnsemble(
        num_encoders=cfg["num_encoders"],
        audio_input_size=cfg["audio_input_size"],
        video_input_size=cfg["video_input_size"],
        text_input_size=cfg["text_input_size"],
        mm_input_size=cfg["mm_input_size"],
        mm_output_sizes=cfg["mm_output_sizes"],
        d_ffn=cfg["d_ffn"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], activation=cfg["activation"],
        mamba_config={"d_state": cfg["d_state"], "d_conv": cfg["d_conv"],
                      "expand": cfg["expand"], "bidirectional": True},
        ensemble_stage=cfg["ensemble_stage"], fusion_type=cfg["fusion_type"],
        num_classes=1,
        diversity_loss_weight=cfg["diversity_loss_weight"],
        use_moe=cfg["use_moe"] in (True, "True", "true"),
        num_experts=cfg["num_experts"], top_k_experts=cfg["top_k_experts"],
        moe_loss_weight=cfg["moe_loss_weight"],
        use_pre_cossm_moe=cfg["use_pre_cossm_moe"] in (True, "True", "true"),
        pre_cossm_num_experts=cfg["pre_cossm_num_experts"],
        pre_cossm_top_k=cfg["pre_cossm_top_k"],
        pre_cossm_moe_loss_weight=cfg["pre_cossm_moe_loss_weight"],
        modality_dropout_rate=0.0,
    )


def perturb(x, sl, mode, ratio=1.0, sigma=0.0, rng=None):
    """Zero or corrupt x[:, :, sl] (B,T,D)."""
    x = x.clone()
    B, T, _ = x.shape
    if mode == "zero_all":
        x[:, :, sl] = 0.0
    elif mode == "zero_segment":
        L = int(T * ratio)
        if L > 0:
            start = rng.integers(0, max(T - L, 1))
            x[:, start:start + L, sl] = 0.0
    elif mode == "zero_random":
        n = int(T * ratio)
        if n > 0:
            idx = torch.from_numpy(rng.choice(T, size=n, replace=False))
            x[:, idx, sl] = 0.0
    elif mode == "noise":
        feat = x[:, :, sl]
        x[:, :, sl] = feat + sigma * feat.std() * torch.randn_like(feat)
    return x


def evaluate(net, loader, device, cond_fn):
    net.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            x = cond_fn(x)
            pred = (net(x, mask) > 0.).int()
            TP += ((pred == 1) & (y == 1)).sum().item()
            FP += ((pred == 1) & (y == 0)).sum().item()
            TN += ((pred == 0) & (y == 0)).sum().item()
            FN += ((pred == 0) & (y == 1)).sum().item()
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (TP + TN) / (TP + FP + TN + FN)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "acc": round(acc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="dir containing config.json and checkpoints/best_model.pt")
    ap.add_argument("--data_dir", default=os.path.join(os.environ.get("DEPM3_DATA_ROOT", "./data"), "dvlog"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ckpt", action="store_true", help="dry run with random weights")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="eval batch size (FiLM missing-flag is batch-aggregated; use 1 for per-sample)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    dataset = cfg.get("dataset", "dvlog")
    sls = modality_slices(dataset)
    device = torch.device(args.device)

    net = build_model(cfg).to(device)
    if not args.no_ckpt:
        sd = torch.load(os.path.join(args.run_dir, "checkpoints", "best_model.pt"),
                        map_location=device)
        net.load_state_dict(sd)

    if dataset == "lmvd":
        loader = get_lmvd_trimodal_dataloader(
            root=os.environ.get("LMVD_DATA_ROOT", os.path.join(os.environ.get("DEPM3_DATA_ROOT", "./data"), "lmvd")),
            fold="test", batch_size=args.batch_size, gender="both", aug=False)
    else:
        loader = get_dvlog_trimodal_dataloader(
            root=args.data_dir, fold="test", batch_size=args.batch_size, gender="both", aug=False)

    rng = np.random.default_rng(args.seed)
    results = {"run_dir": args.run_dir, "dataset": dataset,
               "config_summary": {k: cfg[k] for k in
                                  ["num_encoders", "use_moe", "use_pre_cossm_moe",
                                   "diversity_loss_weight", "modality_dropout_rate"]}}

    # A. Global conditions (Fig.2)
    conds = {
        "full": lambda x: x,
        "wo_audio": lambda x: perturb(x, sls["audio"], "zero_all"),
        "wo_visual": lambda x: perturb(x, sls["visual"], "zero_all"),
        "wo_text": lambda x: perturb(x, sls["text"], "zero_all"),
        "audio_only": lambda x: perturb(perturb(x, sls["visual"], "zero_all"),
                                        sls["text"], "zero_all"),
        "visual_only": lambda x: perturb(perturb(x, sls["audio"], "zero_all"),
                                         sls["text"], "zero_all"),
        "text_only": lambda x: perturb(perturb(x, sls["audio"], "zero_all"),
                                       sls["visual"], "zero_all"),
    }
    results["global"] = {}
    for name, fn in conds.items():
        results["global"][name] = evaluate(net, loader, device, fn)
        print(f"[global] {name}: {results['global'][name]}", flush=True)

    # B. Frame-level local masking
    results["frame_level"] = {}
    for mod in ["audio", "visual"]:
        for mode in ["zero_segment", "zero_random"]:
            for ratio in [0.1, 0.3, 0.5]:
                key = f"{mod}_{mode}_{ratio}"
                fn = (lambda m=mod, mo=mode, r=ratio:
                      lambda x: perturb(x, sls[m], mo, ratio=r, rng=rng))()
                results["frame_level"][key] = evaluate(net, loader, device, fn)
                print(f"[frame] {key}: {results['frame_level'][key]}", flush=True)

    # C. Gaussian corruption (non-zero, evades L1 detector)
    results["noise"] = {}
    for mod in ["audio", "visual"]:
        for sigma in [0.5, 1.0]:
            key = f"{mod}_sigma{sigma}"
            fn = (lambda m=mod, s=sigma: lambda x: perturb(x, sls[m], "noise", sigma=s))()
            results["noise"][key] = evaluate(net, loader, device, fn)
            print(f"[noise] {key}: {results['noise'][key]}", flush=True)

    out = args.out or os.path.join(args.run_dir, "missing_modality_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
