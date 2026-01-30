"""Main script for Multi-Encoder Ensemble training.
End-to-end training of multiple multimodal encoders with ensemble fusion.

This is a SINGLE UNIFIED script that trains everything together.

Usage:
    python main_multiencoder_ensemble.py --num_encoders 3 --fusion_type weighted --epochs 120
"""

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

from models.DepMamba_multiencoder_ensemble import DepMambaMultiEncoderEnsemble
from datasets import get_dvlog_trimodal_dataloader, get_lmvd_trimodal_dataloader

# Default data paths
LMVD_DATA_PATH = "DATAPATH"


CONFIG_PATH = "./config/config_ensemble.yaml"


def parse_args():
    """Parse command line arguments."""
    # Load config file if exists
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(
        description="Train Multi-Encoder Ensemble model for multimodal depression detection."
    )
    
    # Ensemble settings
    parser.add_argument("--num_encoders", type=int, default=3,
                       help="Number of multimodal encoders in ensemble")
    parser.add_argument("--ensemble_stage", type=str, default="late",
                       choices=['early', 'late'],
                       help="Ensemble stage: 'early' (feature fusion) or 'late' (logits ensemble)")
    parser.add_argument("--fusion_type", type=str, default="weighted",
                       help="Type of ensemble fusion. "
                            "For early: average, weighted, attention, concat. "
                            "For late: average, weighted, attention, voting, stacking, gating")
    parser.add_argument("--diversity_loss_weight", type=float, default=0.01,
                       help="Weight for diversity loss")
    
    # MoE settings
    parser.add_argument("--use_moe", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                       help="Whether to use MoE in encoders")
    parser.add_argument("--num_experts", type=int, default=6,
                       help="Number of experts per MoE layer")
    parser.add_argument("--top_k_experts", type=int, default=2,
                       help="Number of top experts to activate")
    parser.add_argument("--moe_loss_weight", type=float, default=0.01,
                       help="Weight for MoE load balancing loss")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=config.get('data_dir', 'DATASET_PATH'))
    parser.add_argument("--dataset", type=str, default="dvlog", choices=['dvlog', 'lmvd'])
    parser.add_argument("--train_gender", type=str, default=config.get('train_gender', 'both'))
    parser.add_argument("--test_gender", type=str, default=config.get('test_gender', 'both'))
    
    # Model arguments
    parser.add_argument("--audio_input_size", type=int, default=25,
                       help="Audio feature dimension (25 for D-Vlog, 128 for LMVD)")
    parser.add_argument("--video_input_size", type=int, default=136,
                       help="Visual feature dimension")
    parser.add_argument("--text_input_size", type=int, default=768,
                       help="Text feature dimension (BERT)")
    parser.add_argument("--mm_input_size", type=int, default=256,
                       help="Common dimension after projection")
    parser.add_argument("--mm_output_sizes", type=int, nargs="+", default=[256, 64],
                       help="Output sizes for encoder layers")
    parser.add_argument("--d_ffn", type=int, default=1024,
                       help="Feed-forward network dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--activation", type=str, default="GELU",
                       help="Activation function")
    
    # Mamba config
    parser.add_argument("--d_state", type=int, default=12,
                       help="Mamba d_state")
    parser.add_argument("--d_conv", type=int, default=4,
                       help="Mamba d_conv")
    parser.add_argument("--expand", type=int, default=4,
                       help="Mamba expand")
    
    # Training arguments
    parser.add_argument("-e", "--epochs", type=int, default=120,
                       help="Number of training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=8e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cos",
                       choices=['cos', 'plateau'],
                       help="Learning rate scheduler")
    parser.add_argument("--early_stopping", type=int, default=1000,
                       help="Early stopping patience (0 to disable)")
    
    # System arguments
    parser.add_argument("-g", "--gpu", type=str, default="0",
                       help="GPU device (e.g., '0' or '1')")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (e.g., 'cuda:0', 'cuda:1'). Overrides --gpu if set.")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Save directory")
    parser.add_argument("--tqdm_able", action="store_true",
                       help="Disable tqdm progress bar")
    parser.add_argument("--train", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                       help="Train the model (True/False)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_iterations", type=int, default=3,
                       help="Number of training iterations")
    parser.add_argument("--if_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Adjust for dataset
    if args.dataset == 'lmvd':
        args.audio_input_size = 128  # VGGish features
    
    return args


def train_epoch(
    net, train_loader, loss_fn, optimizer, device, 
    current_epoch, total_epochs, tqdm_able=False
):
    """Train for one epoch."""
    net.train()
    sample_count = 0
    running_loss = 0.
    running_diversity_loss = 0.
    running_moe_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Epoch {current_epoch+1}/{total_epochs}",
        leave=False, unit="batch", disable=tqdm_able
    ) as pbar:
        for x, y, mask in pbar:
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            
            # Forward pass
            y_pred = net(x, mask, return_diversity_loss=True, return_moe_loss=True)
            
            # Classification loss
            cls_loss = loss_fn(y_pred, y.to(torch.float32))
            
            # Diversity loss
            diversity_loss = net.get_diversity_loss()
            if diversity_loss is None:
                diversity_loss = torch.tensor(0.0, device=device)
            elif not isinstance(diversity_loss, torch.Tensor):
                diversity_loss = torch.tensor(diversity_loss, device=device)
            
            # MoE load balancing loss
            moe_loss = net.get_moe_loss()
            if moe_loss is None:
                moe_loss = torch.tensor(0.0, device=device)
            elif not isinstance(moe_loss, torch.Tensor):
                moe_loss = torch.tensor(moe_loss, device=device)
            moe_loss = moe_loss.to(device)
            
            # Total loss = classification + diversity + MoE
            total_loss = cls_loss + diversity_loss + moe_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += cls_loss.item() * x.shape[0]
            running_diversity_loss += diversity_loss.item() * x.shape[0]
            running_moe_loss += moe_loss.item() * x.shape[0]
            
            # Binary classification
            pred = (y_pred > 0.).int()
            correct_count += (pred == y).sum().item()

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
        "acc": correct_count / sample_count,
    }


def validate(net, val_loader, loss_fn, device, tqdm_able=False):
    """Validate the model."""
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                
                y_pred = net(x, mask, return_diversity_loss=False)
                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                
                # Binary classification
                pred = (y_pred > 0.).int()
                
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

    # Calculate metrics
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
    """Save training results."""
    # Save as JSON
    json_path = os.path.join(save_path, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    csv_path = os.path.join(save_path, "training_history.csv")
    if 'epoch_history' in results and results['epoch_history']:
        fieldnames = results['epoch_history'][0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['epoch_history'])


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.device:
        # Use --device directly (e.g., cuda:1)
        device = torch.device(args.device)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print("="*80)
    print("Multi-Encoder Ensemble + MoE Training")
    print("="*80)
    print(f"Number of encoders: {args.num_encoders}")
    print(f"Ensemble stage: {args.ensemble_stage.upper()}")
    print(f"  - early: Feature fusion → Single classifier")
    print(f"  - late: Individual classifiers → Logits ensemble")
    print(f"Fusion type: {args.fusion_type}")
    print(f"Diversity loss weight: {args.diversity_loss_weight}")
    print(f"--- MoE Settings ---")
    print(f"Use MoE: {args.use_moe}")
    print(f"Number of experts: {args.num_experts}")
    print(f"Top-k experts: {args.top_k_experts}")
    print(f"MoE loss weight: {args.moe_loss_weight}")
    print(f"---")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*80)
    
    # Mamba config
    mamba_config = {
        'd_state': args.d_state,
        'd_conv': args.d_conv,
        'expand': args.expand,
        'bidirectional': True
    }
    
    # Run multiple iterations
    for i_iter in range(args.num_iterations):
        print(f"\n{'#'*80}")
        print(f"# Iteration {i_iter + 1}/{args.num_iterations}")
        print(f"{'#'*80}")
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        moe_str = f"moe{args.num_experts}k{args.top_k_experts}" if args.use_moe else "nomoe"
        save_folder = f"multienc_{args.ensemble_stage}_{args.num_encoders}enc_{moe_str}_{args.fusion_type}_{i_iter}_{timestamp}"
        save_path = os.path.join(args.save_dir, args.dataset, save_folder)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
        
        print(f"Save path: {save_path}")
        
        # Create data loaders
        print("\nLoading data...")
        
        if args.dataset == 'lmvd':
            # LMVD dataset
            data_path = LMVD_DATA_PATH
            print(f"Using LMVD dataset from: {data_path}")
            
            train_loader = get_lmvd_trimodal_dataloader(
                root=data_path,
                fold="train",
                batch_size=args.batch_size,
                gender=args.train_gender,
                aug=True,
            )
            
            val_loader = get_lmvd_trimodal_dataloader(
                root=data_path,
                fold="valid",
                batch_size=args.batch_size,
                gender=args.test_gender,
                aug=False,
            )
            
            test_loader = get_lmvd_trimodal_dataloader(
                root=data_path,
                fold="test",
                batch_size=args.batch_size,
                gender=args.test_gender,
                aug=False,
            )
        else:
            # D-Vlog dataset (default)
            data_path = os.path.join(args.data_dir, args.dataset) if args.dataset not in args.data_dir else args.data_dir
            print(f"Using D-Vlog dataset from: {data_path}")
            
            train_loader = get_dvlog_trimodal_dataloader(
                root=data_path,
                fold="train",
                batch_size=args.batch_size,
                gender=args.train_gender,
                aug=True,
            )
            
            val_loader = get_dvlog_trimodal_dataloader(
                root=data_path,
                fold="valid",
                batch_size=args.batch_size,
                gender=args.test_gender,
                aug=False,
            )
            
            test_loader = get_dvlog_trimodal_dataloader(
                root=data_path,
                fold="test",
                batch_size=args.batch_size,
                gender=args.test_gender,
                aug=False,
            )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Valid samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Create model
        print("\nCreating model...")
        net = DepMambaMultiEncoderEnsemble(
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
            num_classes=1,
            diversity_loss_weight=args.diversity_loss_weight,
            # MoE parameters
            use_moe=args.use_moe,
            num_experts=args.num_experts,
            top_k_experts=args.top_k_experts,
            moe_loss_weight=args.moe_loss_weight,
        )
        
        print(f"Model type: {args.ensemble_stage.upper()} Ensemble + MoE")
        print(f"MoE enabled: {args.use_moe}")
        if args.use_moe:
            print(f"  - {args.num_experts} experts per layer, top-{args.top_k_experts} activated")
        if args.ensemble_stage == 'late':
            print(f"  - Each encoder has its own classifier")
            print(f"  - Logits are ensembled using '{args.fusion_type}' method")
        else:
            print(f"  - Features are fused using '{args.fusion_type}' method")
            print(f"  - Single classifier after fusion")
        
        net = net.to(device)
        
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Save config
        config_dict = vars(args).copy()
        config_dict['mamba_config'] = mamba_config
        config_dict['total_params'] = total_params
        config_dict['iteration'] = i_iter
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Training setup
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            net.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        if args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # Training loop
        if args.train:
            print("\n" + "="*80)
            print("Starting training...")
            print("="*80)
            
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
                
                # Calculate validation score
                val_score = (
                    val_results["acc"] + val_results["precision"] + 
                    val_results["recall"] + val_results["f1"]
                ) / 4.0
                
                # Update scheduler
                if args.lr_scheduler == 'cos':
                    scheduler.step()
                else:
                    scheduler.step(val_score)
                
                # Save best model
                if val_score > best_val_score:
                    best_val_score = val_score
                    epochs_without_improvement = 0
                    torch.save(
                        net.state_dict(),
                        os.path.join(save_path, "checkpoints", "best_model.pt")
                    )
                    np.save(
                        os.path.join(save_path, "checkpoints", "best_confusion_matrix.npy"),
                        val_cm
                    )
                else:
                    epochs_without_improvement += 1
                
                # Record epoch results
                epoch_record = {
                    'epoch': epoch + 1,
                    'timestamp': datetime.now().isoformat(),
                    'train_loss': train_results['loss'],
                    'train_diversity_loss': train_results['diversity_loss'],
                    'train_moe_loss': train_results['moe_loss'],
                    'train_acc': train_results['acc'],
                    'val_loss': val_results['loss'],
                    'val_acc': val_results['acc'],
                    'val_precision': val_results['precision'],
                    'val_recall': val_results['recall'],
                    'val_f1': val_results['f1'],
                    'val_score': val_score,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                
                # Get ensemble weights if available
                weights = net.get_ensemble_weights()
                if weights is not None:
                    for i, w in enumerate(weights.detach().cpu().numpy()):
                        epoch_record[f'weight_enc{i}'] = float(w)
                
                epoch_history.append(epoch_record)
                
                # Print progress
                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train - Loss: {train_results['loss']:.4f}, "
                      f"Div: {train_results['diversity_loss']:.4f}, "
                      f"MoE: {train_results['moe_loss']:.4f}, "
                      f"Acc: {train_results['acc']:.4f}")
                print(f"  Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['acc']:.4f}")
                print(f"          P/R/F1: {val_results['precision']:.4f}/"
                      f"{val_results['recall']:.4f}/{val_results['f1']:.4f}")
                print(f"          Score: {val_score:.4f}, Best: {best_val_score:.4f}")
                
                if weights is not None:
                    weight_str = ", ".join([f"{w:.3f}" for w in weights.detach().cpu().numpy()])
                    print(f"          Ensemble weights: [{weight_str}]")
                
                if val_score >= best_val_score:
                    print(f"  *** New best model saved ***")
                
                # Early stopping
                if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
                    print(f"\nEarly stopping triggered after {args.early_stopping} epochs without improvement.")
                    break
            
            # Save training summary
            training_summary = {
                'iteration': i_iter,
                'num_encoders': args.num_encoders,
                'ensemble_stage': args.ensemble_stage,
                'fusion_type': args.fusion_type,
                'use_moe': args.use_moe,
                'num_experts': args.num_experts,
                'top_k_experts': args.top_k_experts,
                'total_epochs_trained': epoch + 1,
                'best_val_score': best_val_score,
                'config': config_dict,
                'epoch_history': epoch_history,
                'training_completed': datetime.now().isoformat(),
            }
            save_results(training_summary, save_path)
        
        # Testing
        print("\n" + "="*80)
        print("Testing...")
        print("="*80)
        
        # Load best model
        net.load_state_dict(
            torch.load(os.path.join(save_path, "checkpoints", "best_model.pt"), map_location=device)
        )
        
        test_results, test_cm = validate(net, test_loader, loss_fn, device, args.tqdm_able)
        
        print(f"\nTest Results:")
        print(f"  Ensemble Stage: {args.ensemble_stage}")
        print(f"  MoE enabled: {args.use_moe}")
        if args.use_moe:
            print(f"  Experts: {args.num_experts}, Top-K: {args.top_k_experts}")
        print(f"  Accuracy: {test_results['acc']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall: {test_results['recall']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")
        print(f"\nConfusion Matrix:\n{test_cm}")
        
        # Get final ensemble weights
        weights = net.get_ensemble_weights()
        if weights is not None:
            print(f"\nFinal Ensemble Weights:")
            for i, w in enumerate(weights.detach().cpu().numpy()):
                print(f"  Encoder {i+1}: {w:.4f}")
        
        # Get ensemble info for late ensemble
        ensemble_info = net.get_ensemble_info()
        if ensemble_info and 'individual_logits' in ensemble_info:
            print(f"\nLate Ensemble - Individual Encoder Outputs Available")
            print(f"  (Use get_individual_predictions() for detailed analysis)")
        
        # Save test results
        test_avg_score = (
            test_results['acc'] + test_results['precision'] + 
            test_results['recall'] + test_results['f1']
        ) / 4.0
        
        with open(os.path.join(save_path, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        np.save(os.path.join(save_path, "test_confusion_matrix.npy"), test_cm)
        
        # Save as text
        result_txt_path = os.path.join(
            args.save_dir, args.dataset,
            f"{args.dataset}_multienc_{args.ensemble_stage}_{args.num_encoders}enc_{moe_str}_{args.fusion_type}_{i_iter}_{timestamp}.txt"
        )
        with open(result_txt_path, 'w') as f:
            f.write(f"Multi-Encoder Ensemble + MoE Results\n")
            f.write(f"=====================================\n")
            f.write(f"Ensemble Stage: {args.ensemble_stage}\n")
            f.write(f"Number of Encoders: {args.num_encoders}\n")
            f.write(f"Fusion Type: {args.fusion_type}\n")
            f.write(f"\nMoE Settings:\n")
            f.write(f"  Use MoE: {args.use_moe}\n")
            f.write(f"  Num Experts: {args.num_experts}\n")
            f.write(f"  Top-K Experts: {args.top_k_experts}\n")
            f.write(f"  MoE Loss Weight: {args.moe_loss_weight}\n")
            f.write(f"\nResults:\n")
            f.write(f"Accuracy: {test_results['acc']:.4f}\n")
            f.write(f"Precision: {test_results['precision']:.4f}\n")
            f.write(f"Recall: {test_results['recall']:.4f}\n")
            f.write(f"F1: {test_results['f1']:.4f}\n")
            f.write(f"Average Score: {test_avg_score:.4f}\n")
            if weights is not None:
                f.write(f"\nEnsemble Weights:\n")
                for i, w in enumerate(weights.detach().cpu().numpy()):
                    f.write(f"  Encoder {i+1}: {w:.4f}\n")
        
        print(f"\nResults saved to {result_txt_path}")
    
    print("\n" + "="*80)
    print("All iterations completed!")
    print("="*80)


if __name__ == '__main__':
    main()
