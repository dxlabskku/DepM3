import argparse
import os
import yaml
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from models.DepMamba_MUD3_user_level import DepMambaMUD3UserLevel
from datasets import get_mud3_dataloader

MUD3_DATA_PATH = os.environ.get("MUD3_DATA_ROOT", "./data/MUD3")


def _get_module(net):
    """Unwrap DataParallel to access underlying model methods."""
    return net.module if isinstance(net, nn.DataParallel) else net


CONFIG_PATH = "./config/config_ensemble.yaml"


def parse_args():
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train MUD3 User-Level model for depression detection."
    )

    # Ensemble settings
    parser.add_argument("--num_encoders", type=int, default=3)
    parser.add_argument("--ensemble_stage", type=str, default="late",
                       choices=['early', 'late'])
    parser.add_argument("--fusion_type", type=str, default="weighted")
    parser.add_argument("--diversity_loss_weight", type=float, default=0.01)

    # MoE settings (Post-CoSSM)
    parser.add_argument("--use_moe", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--top_k_experts", type=int, default=3)
    parser.add_argument("--moe_loss_weight", type=float, default=0.01)

    # Pre-CoSSM FiLM MoE settings
    parser.add_argument("--use_pre_cossm_moe", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
    parser.add_argument("--pre_cossm_num_experts", type=int, default=6)
    parser.add_argument("--pre_cossm_top_k", type=int, default=2)
    parser.add_argument("--pre_cossm_moe_loss_weight", type=float, default=0.01)

    # Modality dropout
    parser.add_argument("--modality_dropout_rate", type=float, default=0.15)

    # Data arguments
    parser.add_argument("--data_dir", type=str, default=MUD3_DATA_PATH)

    # Video encoder model arguments
    parser.add_argument("--audio_input_size", type=int, default=25)
    parser.add_argument("--video_input_size", type=int, default=136)
    parser.add_argument("--text_input_size", type=int, default=768)
    parser.add_argument("--mm_input_size", type=int, default=256)
    parser.add_argument("--mm_output_sizes", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="GELU")

    # Mamba config
    parser.add_argument("--d_state", type=int, default=12)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=4)

    # User aggregator arguments
    parser.add_argument("--user_aggregator", type=str, default="mamba",
                       choices=['gru', 'mamba', 'mean', 'attention', 'transformer'],
                       help="User-level aggregator: 'gru' (biGRU) or 'mamba' (BiMamba)")
    parser.add_argument("--agg_hidden_dim", type=int, default=128,
                       help="Hidden dimension for user aggregator")
    parser.add_argument("--agg_num_layers", type=int, default=2,
                       help="Number of layers in user aggregator")
    parser.add_argument("--agg_dropout", type=float, default=0.1)
    parser.add_argument("--chunk_size", type=int, default=64,
                       help="Chunk size for video encoder (memory efficiency)")

    # Training arguments
    parser.add_argument("-e", "--epochs", type=int, default=120)
    parser.add_argument("-bs", "--batch_size", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="cos",
                       choices=['cos', 'plateau'])
    parser.add_argument("--early_stopping", type=int, default=30)

    # System arguments
    parser.add_argument("-g", "--gpu", type=str, default="0",
                       help="GPU devices (e.g., '0', '0,1' for multi-GPU)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--tqdm_able", action="store_true")
    parser.add_argument("--train", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--aug", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)

    args = parser.parse_args()
    return args


def train_epoch(
    net, train_loader, loss_fn, optimizer, device,
    current_epoch, total_epochs, tqdm_able=False
):
    net.train()
    sample_count = 0
    running_loss = 0.
    running_diversity_loss = 0.
    running_moe_loss = 0.
    running_pre_moe_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Epoch {current_epoch+1}/{total_epochs}",
        leave=False, unit="batch", disable=tqdm_able
    ) as pbar:
        for visual, acoustic, y, mask in pbar:
            visual = visual.to(device)
            acoustic = acoustic.to(device)
            y = y.to(device).unsqueeze(1).float()
            mask = mask.to(device)

            fwd = _get_module(net) if (
                isinstance(net, nn.DataParallel)
                and visual.shape[0] < len(net.device_ids)
            ) else net
            y_pred = fwd(visual, acoustic, mask,
                        return_diversity_loss=True, return_moe_loss=True)

            cls_loss = loss_fn(y_pred, y)

            m = _get_module(net)

            diversity_loss = m.get_diversity_loss()
            if diversity_loss is None:
                diversity_loss = torch.tensor(0.0, device=device)
            elif not isinstance(diversity_loss, torch.Tensor):
                diversity_loss = torch.tensor(diversity_loss, device=device)

            moe_loss = m.get_moe_loss()
            if moe_loss is None:
                moe_loss = torch.tensor(0.0, device=device)
            elif not isinstance(moe_loss, torch.Tensor):
                moe_loss = torch.tensor(moe_loss, device=device)
            moe_loss = moe_loss.to(device)

            pre_moe_loss = m.get_pre_moe_loss()
            if pre_moe_loss is None:
                pre_moe_loss = torch.tensor(0.0, device=device)
            elif not isinstance(pre_moe_loss, torch.Tensor):
                pre_moe_loss = torch.tensor(pre_moe_loss, device=device)
            pre_moe_loss = pre_moe_loss.to(device)

            total_loss = cls_loss + diversity_loss + moe_loss + pre_moe_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_size = visual.shape[0]
            sample_count += batch_size
            running_loss += cls_loss.item() * batch_size
            running_diversity_loss += diversity_loss.item() * batch_size
            running_moe_loss += moe_loss.item() * batch_size
            running_pre_moe_loss += pre_moe_loss.item() * batch_size

            pred = (y_pred > 0.).int()
            correct_count += (pred == y.int()).sum().item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "div": running_diversity_loss / sample_count,
                "moe": running_moe_loss / sample_count,
                "acc": correct_count / sample_count,
            })

    return {
        "loss": running_loss / sample_count,
        "diversity_loss": running_diversity_loss / sample_count,
        "moe_loss": running_moe_loss / sample_count,
        "pre_moe_loss": running_pre_moe_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def validate(net, val_loader, loss_fn, device, tqdm_able=False):
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for visual, acoustic, y, mask in pbar:
                visual = visual.to(device)
                acoustic = acoustic.to(device)
                y = y.to(device).unsqueeze(1).float()
                mask = mask.to(device)

                fwd = _get_module(net) if (
                    isinstance(net, nn.DataParallel)
                    and visual.shape[0] < len(net.device_ids)
                ) else net
                y_pred = fwd(visual, acoustic, mask,
                            return_diversity_loss=False, return_moe_loss=False)
                loss = loss_fn(y_pred, y)

                batch_size = visual.shape[0]
                sample_count += batch_size
                running_loss += loss.item() * batch_size

                pred = (y_pred > 0.).int()
                y_int = y.int()

                TP += torch.sum((pred == 1) & (y_int == 1)).item()
                FP += torch.sum((pred == 1) & (y_int == 0)).item()
                TN += torch.sum((pred == 0) & (y_int == 0)).item()
                FN += torch.sum((pred == 0) & (y_int == 1)).item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    return {
        "loss": running_loss / sample_count,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, confusion_matrix


def save_results(results, save_path):
    json_path = os.path.join(save_path, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(save_path, "training_history.csv")
    if 'epoch_history' in results and results['epoch_history']:
        fieldnames = results['epoch_history'][0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['epoch_history'])


def main():
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.device:
        device = torch.device(args.device)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_ids = [int(g) for g in args.gpu.split(',')]
    use_multi_gpu = len(gpu_ids) > 1

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("=" * 80)
    print("MUD3 User-Level Depression Detection")
    print("=" * 80)
    print(f"User Aggregator: {args.user_aggregator.upper()}")
    print(f"  - Hidden dim: {args.agg_hidden_dim}, Layers: {args.agg_num_layers}")
    print(f"  - Chunk size: {args.chunk_size}")
    print(f"Video Encoder: {args.num_encoders} encoders, {args.ensemble_stage} ensemble")
    print(f"  - Fusion: {args.fusion_type}")
    print(f"  - Post-CoSSM MoE: {args.use_moe} ({args.num_experts} experts, top-{args.top_k_experts})")
    print(f"  - Pre-CoSSM FiLM MoE: {args.use_pre_cossm_moe} ({args.pre_cossm_num_experts} experts, top-{args.pre_cossm_top_k})")
    print(f"  - Modality dropout: {args.modality_dropout_rate}")
    print(f"Training: epochs={args.epochs}, bs={args.batch_size}, lr={args.learning_rate}")
    print("=" * 80)

    mamba_config = {
        'd_state': args.d_state,
        'd_conv': args.d_conv,
        'expand': args.expand,
        'bidirectional': True,
    }

    for i_iter in range(args.num_iterations):
        print(f"\n{'#' * 80}")
        print(f"# Iteration {i_iter + 1}/{args.num_iterations}")
        print(f"{'#' * 80}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moe_str = f"moe{args.num_experts}k{args.top_k_experts}" if args.use_moe else "nomoe"
        pre_moe_str = f"_film{args.pre_cossm_num_experts}k{args.pre_cossm_top_k}" if args.use_pre_cossm_moe else ""
        agg_str = f"_{args.user_aggregator}{args.agg_hidden_dim}L{args.agg_num_layers}"
        save_folder = (
            f"mud3_{args.ensemble_stage}_{args.num_encoders}enc"
            f"_{moe_str}{pre_moe_str}{agg_str}"
            f"_{args.fusion_type}_{i_iter}_{timestamp}"
        )
        save_path = os.path.join(args.save_dir, "mud3", save_folder)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

        print(f"Save path: {save_path}")

        # Data loaders
        print("\nLoading MUD3 data...")
        train_loader = get_mud3_dataloader(
            root=args.data_dir, fold="train",
            batch_size=args.batch_size, aug=args.aug,
            drop_last=use_multi_gpu,
        )
        val_loader = get_mud3_dataloader(
            root=args.data_dir, fold="val",
            batch_size=args.batch_size, aug=False,
        )
        test_loader = get_mud3_dataloader(
            root=args.data_dir, fold="test",
            batch_size=args.batch_size, aug=False,
        )
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        # Model
        print("\nCreating model...")
        net = DepMambaMUD3UserLevel(
            num_encoders=args.num_encoders,
            audio_input_size=args.audio_input_size,
            video_input_size=args.video_input_size,
            text_input_size=args.text_input_size,
            mm_input_size=args.mm_input_size,
            mm_output_sizes=args.mm_output_sizes,
            d_ffn=args.d_ffn,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            mamba_config=mamba_config,
            ensemble_stage=args.ensemble_stage,
            fusion_type=args.fusion_type,
            diversity_loss_weight=args.diversity_loss_weight,
            use_moe=args.use_moe,
            num_experts=args.num_experts,
            top_k_experts=args.top_k_experts,
            moe_loss_weight=args.moe_loss_weight,
            use_pre_cossm_moe=args.use_pre_cossm_moe,
            pre_cossm_num_experts=args.pre_cossm_num_experts,
            pre_cossm_top_k=args.pre_cossm_top_k,
            pre_cossm_moe_loss_weight=args.pre_cossm_moe_loss_weight,
            modality_dropout_rate=args.modality_dropout_rate,
            user_aggregator=args.user_aggregator,
            agg_hidden_dim=args.agg_hidden_dim,
            agg_num_layers=args.agg_num_layers,
            agg_dropout=args.agg_dropout,
            chunk_size=args.chunk_size,
        ).to(device)

        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        num_gpus = torch.cuda.device_count()
        if use_multi_gpu and num_gpus >= len(gpu_ids):
            dp_device_ids = list(range(len(gpu_ids)))
            print(f"Using DataParallel on {len(gpu_ids)} GPUs (physical: {gpu_ids})")
            if args.batch_size < len(gpu_ids):
                print(f"[WARNING] batch_size({args.batch_size}) < num_gpus({len(gpu_ids)}). "
                      f"Small batches will fallback to single GPU. "
                      f"Recommend batch_size >= {len(gpu_ids)}.")
            net = nn.DataParallel(net, device_ids=dp_device_ids)
            net_module = net.module
        else:
            net_module = net

        config_dict = vars(args).copy()
        config_dict['mamba_config'] = mamba_config
        config_dict['total_params'] = total_params
        config_dict['iteration'] = i_iter
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        if args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        if args.train:
            print("\n" + "=" * 80)
            print("Starting training...")
            print("=" * 80)

            best_val_score = -1.0
            epochs_without_improvement = 0
            epoch_history = []

            for epoch in range(args.epochs):
                train_results = train_epoch(
                    net, train_loader, loss_fn, optimizer,
                    device, epoch, args.epochs, args.tqdm_able
                )

                val_results, val_cm = validate(
                    net, val_loader, loss_fn, device, args.tqdm_able
                )

                val_score =  val_results["f1"]
                # (
                #     val_results["acc"] + val_results["precision"] +
                #     val_results["recall"] + val_results["f1"]
                # ) / 4.0

                if args.lr_scheduler == 'cos':
                    scheduler.step()
                else:
                    scheduler.step(val_score)

                if val_score > best_val_score:
                    best_val_score = val_score
                    epochs_without_improvement = 0
                    torch.save(
                        net_module.state_dict(),
                        os.path.join(save_path, "checkpoints", "best_model.pt")
                    )
                    np.save(
                        os.path.join(save_path, "checkpoints", "best_confusion_matrix.npy"),
                        val_cm
                    )
                else:
                    epochs_without_improvement += 1

                epoch_record = {
                    'epoch': epoch + 1,
                    'timestamp': datetime.now().isoformat(),
                    'train_loss': train_results['loss'],
                    'train_diversity_loss': train_results['diversity_loss'],
                    'train_moe_loss': train_results['moe_loss'],
                    'train_pre_moe_loss': train_results['pre_moe_loss'],
                    'train_acc': train_results['acc'],
                    'val_loss': val_results['loss'],
                    'val_acc': val_results['acc'],
                    'val_precision': val_results['precision'],
                    'val_recall': val_results['recall'],
                    'val_f1': val_results['f1'],
                    'val_score': val_score,
                    'lr': optimizer.param_groups[0]['lr'],
                }

                weights = net.get_ensemble_weights()
                if weights is not None:
                    for i, w in enumerate(weights.detach().cpu().numpy()):
                        epoch_record[f'weight_enc{i}'] = float(w)

                epoch_history.append(epoch_record)

                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train - Loss: {train_results['loss']:.4f}, "
                      f"Div: {train_results['diversity_loss']:.4f}, "
                      f"MoE: {train_results['moe_loss']:.4f}, "
                      f"PreMoE: {train_results['pre_moe_loss']:.4f}, "
                      f"Acc: {train_results['acc']:.4f}")
                print(f"  Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['acc']:.4f}")
                print(f"          P/R/F1: {val_results['precision']:.4f}/"
                      f"{val_results['recall']:.4f}/{val_results['f1']:.4f}")
                print(f"          Score: {val_score:.4f}, Best: {best_val_score:.4f}")

                if weights is not None:
                    weight_str = ", ".join([f"{w:.3f}" for w in weights.detach().cpu().numpy()])
                    print(f"          Ensemble weights: [{weight_str}]")

                if val_score >= best_val_score:
                    print("  *** New best model saved ***")

                if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
                    print(f"\nEarly stopping after {args.early_stopping} epochs without improvement.")
                    break

            training_summary = {
                'iteration': i_iter,
                'user_aggregator': args.user_aggregator,
                'agg_hidden_dim': args.agg_hidden_dim,
                'agg_num_layers': args.agg_num_layers,
                'num_encoders': args.num_encoders,
                'ensemble_stage': args.ensemble_stage,
                'fusion_type': args.fusion_type,
                'use_moe': args.use_moe,
                'use_pre_cossm_moe': args.use_pre_cossm_moe,
                'total_epochs_trained': epoch + 1,
                'best_val_score': best_val_score,
                'config': config_dict,
                'epoch_history': epoch_history,
                'training_completed': datetime.now().isoformat(),
            }
            save_results(training_summary, save_path)

        # Testing
        print("\n" + "=" * 80)
        print("Testing...")
        print("=" * 80)

        net_module.load_state_dict(
            torch.load(os.path.join(save_path, "checkpoints", "best_model.pt"), map_location=device)
        )

        test_results, test_cm = validate(net, test_loader, loss_fn, device, args.tqdm_able)

        print(f"\nTest Results:")
        print(f"  User Aggregator: {args.user_aggregator}")
        print(f"  Accuracy:  {test_results['acc']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1:        {test_results['f1']:.4f}")
        print(f"\nConfusion Matrix:\n{test_cm}")

        test_avg_score = (
            test_results['acc'] + test_results['precision'] +
            test_results['recall'] + test_results['f1']
        ) / 4.0

        with open(os.path.join(save_path, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
        np.save(os.path.join(save_path, "test_confusion_matrix.npy"), test_cm)

        result_txt_path = os.path.join(
            args.save_dir, "mud3",
            f"mud3_{args.user_aggregator}_{args.ensemble_stage}_{args.num_encoders}enc_{i_iter}_{timestamp}.txt"
        )
        with open(result_txt_path, 'w') as f:
            f.write(f"MUD3 User-Level Depression Detection Results\n")
            f.write(f"=============================================\n")
            f.write(f"User Aggregator: {args.user_aggregator}\n")
            f.write(f"  Hidden dim: {args.agg_hidden_dim}, Layers: {args.agg_num_layers}\n")
            f.write(f"Video Encoder: {args.num_encoders} encoders, {args.ensemble_stage}\n")
            f.write(f"  Fusion: {args.fusion_type}\n")
            f.write(f"  MoE: {args.use_moe}, Experts: {args.num_experts}, Top-K: {args.top_k_experts}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Accuracy:  {test_results['acc']:.4f}\n")
            f.write(f"  Precision: {test_results['precision']:.4f}\n")
            f.write(f"  Recall:    {test_results['recall']:.4f}\n")
            f.write(f"  F1:        {test_results['f1']:.4f}\n")
            f.write(f"  Avg Score: {test_avg_score:.4f}\n")

        print(f"\nResults saved to {result_txt_path}")

    print("\n" + "=" * 80)
    print("All iterations completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
