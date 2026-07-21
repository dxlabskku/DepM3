"""Measure params / FLOPs / inference latency / peak memory for DepM3 variants.

Addresses reviewer zKDH: "no FLOPs or inference latency analysis is provided."

Configs measured (all D-Vlog dims: audio 25, visual 136, text 768):
  1. backbone_single : N=1, no MoE           (~ DepMamba trimodal equivalent)
  2. depm3_single    : N=1, Pre+Post MoE     (DepM3 w/o ensemble)
  3. ensemble_only   : N=3, no MoE           (plain 3x ensemble)
  4. depm3_full      : N=3, Pre+Post MoE     (full DepM3)

FLOPs are measured with torch.profiler (with_flops=True). Note: this counts
matmul/conv FLOPs and misses the custom selective-scan CUDA kernel, so SSM scan
FLOPs are under-counted for ALL configs equally; relative comparisons hold.
"""
import argparse, json, sys, time, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "post_level"))

import torch
from models.DepMamba_multiencoder_ensemble import DepMambaMultiEncoderEnsemble

MAMBA_CONFIG = {"d_state": 12, "d_conv": 4, "expand": 4, "bidirectional": True}


def build(num_encoders, use_moe):
    return DepMambaMultiEncoderEnsemble(
        num_encoders=num_encoders,
        audio_input_size=25, video_input_size=136, text_input_size=768,
        mm_input_size=256, mm_output_sizes=[256, 64], d_ffn=1024,
        num_layers=2, dropout=0.1, activation="GELU",
        mamba_config=MAMBA_CONFIG,
        ensemble_stage="late", fusion_type="attention", num_classes=1,
        diversity_loss_weight=0.01,
        use_moe=use_moe, num_experts=6, top_k_experts=2, moe_loss_weight=0.01,  # paper checkpoint config
        use_pre_cossm_moe=use_moe, pre_cossm_num_experts=4, pre_cossm_top_k=2,  # paper checkpoint config
        pre_cossm_moe_loss_weight=0.01,
        modality_dropout_rate=0.0,
    )


CONFIGS = {
    "backbone_single(~DepMamba)": dict(num_encoders=1, use_moe=False),
    "depm3_single(w/o ens)":      dict(num_encoders=1, use_moe=True),
    "ensemble_only(3x, no MoE)":  dict(num_encoders=3, use_moe=False),
    "depm3_full(N=3)":            dict(num_encoders=3, use_moe=True),
}


def measure(name, cfg, device, seq_lens, n_warmup=5, n_iter=20):
    torch.manual_seed(0)
    net = build(**cfg).to(device).eval()
    n_params = sum(p.numel() for p in net.parameters())
    out = {"params_M": n_params / 1e6}

    for T in seq_lens:
        x = torch.randn(1, T, 136 + 25 + 768, device=device)
        mask = torch.ones(1, T, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                net(x, mask)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            for _ in range(n_iter):
                net(x, mask)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) / n_iter * 1000
            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

            # FLOPs via profiler (matmul/conv ops; selective-scan kernel not counted)
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         with_flops=True) as prof:
                net(x, mask)
            flops = sum(e.flops for e in prof.key_averages() if e.flops)

        out[f"T{T}"] = {
            "latency_ms": round(latency_ms, 2),
            "peak_mem_MB": round(peak_mb, 1),
            "gflops(profiler)": round(flops / 1e9, 2),
        }
    del net
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficiency_results.json"))
    args = ap.parse_args()

    results = {"device": args.device,
               "gpu": torch.cuda.get_device_name(args.device),
               "note": "batch=1 inference, fp32; profiler FLOPs exclude selective-scan kernel"}
    for name, cfg in CONFIGS.items():
        print(f"Measuring {name} ...", flush=True)
        results[name] = measure(name, cfg, args.device, args.seq_lens)
        print(json.dumps(results[name], indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
