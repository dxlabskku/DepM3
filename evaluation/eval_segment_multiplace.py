"""Segment-masking evaluation averaged over multiple mask placements (5 by default)."""
import argparse, json, sys, os
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_missing import modality_slices, perturb, evaluate, build_model  # noqa
from datasets import get_dvlog_trimodal_dataloader  # noqa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=os.path.join(os.environ.get("DEPM3_DATA_ROOT", "./data"), "dvlog"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_place", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import json as _json
    cfg = _json.load(open(os.path.join(args.run_dir, "config.json")))
    dataset = cfg.get("dataset", "dvlog")
    sls = modality_slices(dataset)
    device = torch.device(args.device)
    net = build_model(cfg).to(device)
    sd = torch.load(os.path.join(args.run_dir, "checkpoints", "best_model.pt"), map_location=device)
    net.load_state_dict(sd)
    loader = get_dvlog_trimodal_dataloader(root=args.data_dir, fold="test",
                                           batch_size=args.batch_size, gender="both", aug=False)
    results = {}
    for mod in ("audio", "visual"):
        for ratio in (0.1, 0.3, 0.5):
            f1s = []
            for seed in range(args.n_place):
                fn = (lambda m=mod, r=ratio, s=seed: lambda x: perturb(
                    x, sls[m], "zero_segment", ratio=r, rng=np.random.default_rng(s)))()
                m_ = evaluate(net, loader, device, fn)
                f1s.append(m_["f1"])
            key = f"{mod}_segment_{ratio}"
            results[key] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                            "f1_all": [round(float(v), 4) for v in f1s]}
            print(key, round(np.mean(f1s)*100, 2), "+-", round(np.std(f1s)*100, 2), flush=True)
        for sigma in (0.5, 1.0):
            f1s = []
            for seed in range(args.n_place):
                torch.manual_seed(seed)
                fn = (lambda m=mod, s=sigma: lambda x: perturb(x, sls[m], "noise", sigma=s))()
                f1s.append(evaluate(net, loader, device, fn)["f1"])
            key = f"{mod}_noise_{sigma}"
            results[key] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            print(key, round(np.mean(f1s)*100, 2), "+-", round(np.std(f1s)*100, 2), flush=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("saved", args.out)

if __name__ == "__main__":
    main()
