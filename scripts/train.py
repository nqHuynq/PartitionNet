import os
import time
import math
import argparse
import json
import csv
import shutil
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from partitionnet.config import TrainingConfig, ModelConfig, LossConfig
from partitionnet.models.partitionnet import PartitionNetEq
from partitionnet.models.loss import StableSymmetricBipolarLoss
from partitionnet.data.dataset import NumericRAMDataset
from partitionnet.data.sampler import TrainChunkBasedDynamicSampler
from partitionnet.data.collate import ChunkCollator
from partitionnet.utils.memory import (
    set_seed,
    trim_memory,
    get_ram_usage,
    clean_state_dict,
    resolve_input_file
)
from partitionnet.utils.metrics import compute_metrics
from partitionnet.utils.invariance import run_permutation_test


def evaluate(model, loader, device, collator):
    model.eval()
    total_samples = len(loader.dataset)
    logits_arr = np.empty((total_samples, 4), dtype=np.float32)
    labels_arr = np.empty((total_samples, 4), dtype=np.float32)
    ptr = 0

    with torch.no_grad():
        for inputs, labels in loader:
            bs = labels.size(0)
            inputs = inputs.to(device, non_blocking=True)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(inputs)
            logits_arr[ptr: ptr + bs] = out.cpu().float().numpy()
            labels_arr[ptr: ptr + bs] = labels.numpy()
            ptr += bs

    exact_match, ham, topo_acc, report = compute_metrics(logits_arr, labels_arr)
    return exact_match, ham, topo_acc, report, logits_arr, labels_arr


def parse_args():
    parser = argparse.ArgumentParser(description="Train PartitionNet-Eq model")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing HDF5 datasets")
    parser.add_argument("--num-train-parts", type=int, default=8, help="Number of training parts")
    parser.add_argument("--epochs", type=int, default=35, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Max train batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate")
    parser.add_argument("--accum-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=9999, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainingConfig()
    
    if args.data_dir:
        cfg.base_h5_path = args.data_dir
    if args.epochs:
        cfg.max_epochs = args.epochs
        cfg.max_epochs_this_run = args.epochs
    if args.batch_size:
        cfg.max_batch_size_train = args.batch_size
    if args.lr:
        cfg.max_lr = args.lr
    if args.accum_steps:
        cfg.accum_steps = args.accum_steps
    if args.patience:
        cfg.patience = args.patience
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed:
        cfg.seed = args.seed
        
    cfg.resolve_paths()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_files = [os.path.join(cfg.base_h5_path, f"iqnet_train_part{i}.h5") for i in range(1, cfg.num_train_parts + 1)]
    val_file = os.path.join(cfg.base_h5_path, "iqnet_val.h5")
    test_file = os.path.join(cfg.base_h5_path, "iqnet_test.h5")

    print(f"Device: {device} | Base Data Path: {cfg.base_h5_path}")
    print(f"Loading training and validation datasets...")

    train_collator = ChunkCollator(
        max_batch_size=cfg.max_batch_size_train,
        max_chunks_limit=cfg.max_chunks_limit,
        window_size=cfg.window_size,
        chunk_buckets=cfg.chunk_buckets
    )
    eval_collator = ChunkCollator(
        max_batch_size=cfg.max_batch_size_eval,
        max_chunks_limit=cfg.max_chunks_limit,
        window_size=cfg.window_size,
        chunk_buckets=cfg.chunk_buckets
    )

    train_dataset = NumericRAMDataset(train_files)
    train_sampler = TrainChunkBasedDynamicSampler(
        train_dataset,
        max_chunks=cfg.max_chunks_per_batch_train,
        max_batch=cfg.max_batch_size_train,
        window_size=cfg.window_size,
        max_chunks_limit=cfg.max_chunks_limit,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=train_collator
    )

    val_dataset = NumericRAMDataset([val_file])
    val_sampler = TrainChunkBasedDynamicSampler(
        val_dataset,
        max_chunks=cfg.max_chunks_per_batch_eval,
        max_batch=cfg.max_batch_size_eval,
        window_size=cfg.window_size,
        max_chunks_limit=cfg.max_chunks_limit,
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=cfg.num_workers,
        collate_fn=eval_collator
    )

    print(f"Loaded Train: {len(train_dataset)} | Val: {len(val_dataset)} | RAM: {get_ram_usage():.1f}%")

    model = PartitionNetEq(window=cfg.window_size).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = StableSymmetricBipolarLoss(lambda_penalty=0.5, temperature=3.0, lambda_pair=0.5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.max_lr,
        weight_decay=cfg.weight_decay
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
    total_steps = steps_per_epoch * cfg.max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        total_steps=total_steps,
        pct_start=cfg.pct_start
    )
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    save_checkpoint_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    save_best_model_path = os.path.join(cfg.output_dir, cfg.best_model_name)
    save_log_path = os.path.join(cfg.output_dir, cfg.log_name)
    save_metrics_path = os.path.join(cfg.output_dir, cfg.metrics_name)
    save_preds_path = os.path.join(cfg.output_dir, cfg.preds_name)

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 1
    epochs_no_improve = 0

    if args.resume and os.path.exists(args.resume):
        chkpt = torch.load(args.resume, map_location=device)
        model_core = model.module if isinstance(model, nn.DataParallel) else model
        model_core.load_state_dict(clean_state_dict(chkpt['model_state']))
        optimizer.load_state_dict(chkpt['optimizer_state'])
        scheduler.load_state_dict(chkpt['scheduler_state'])
        start_epoch = chkpt['epoch'] + 1
        best_val_acc = chkpt.get('best_val_acc', 0.0)
        best_epoch = chkpt.get('best_epoch', start_epoch - 1)
        epochs_no_improve = chkpt.get('epochs_no_improve', 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (Best: {best_val_acc*100:.2f}%)")

    # CSV Logger
    write_header = not os.path.exists(save_log_path)
    with open(save_log_path, mode='a' if not write_header else 'w', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Epoch', 'Time', 'LR', 'RAM_Pct', 'Tr_Loss', 'Tr_Margin', 'Tr_Penalty', 'Va_Exact', 'Va_Ham', 'Va_TopoAcc'])

    for ep in range(start_epoch, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        tr_loss_sum, tr_margin_sum, tr_penalty_sum, tr_total = 0.0, 0.0, 0.0, 0
        current_lr = scheduler.get_last_lr()[0]

        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                out, pair_out = model(inputs, return_pairs=True)
                loss, margin_loss, penalty_loss = criterion(out, labels, pair_out)
                loss_scaled = loss / cfg.accum_steps

            scaler.scale(loss_scaled).backward()

            if (idx + 1) % cfg.accum_steps == 0 or (idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            bs = labels.size(0)
            tr_loss_sum += loss.item() * bs
            tr_margin_sum += margin_loss.item() * bs
            tr_penalty_sum += penalty_loss.item() * bs
            tr_total += bs

        # Validation
        val_em, val_ham, val_acc, _ , _, _ = evaluate(model, val_loader, device, eval_collator)
        ep_time = time.time() - t0
        curr_ram = get_ram_usage()

        print(
            f"Ep {ep:02d}/{cfg.max_epochs} | Time: {ep_time:.1f}s | LR: {current_lr:.2e} | "
            f"Tr: {tr_loss_sum/tr_total:.4f} | Va_Acc: {val_acc*100:.2f}% | Best: {best_val_acc*100:.2f}%"
        )

        with open(save_log_path, mode='a', newline='') as f:
            csv.writer(f).writerow([
                ep, f"{ep_time:.1f}", f"{current_lr:.6f}", f"{curr_ram:.1f}",
                f"{tr_loss_sum/tr_total:.4f}", f"{tr_margin_sum/tr_total:.4f}", f"{tr_penalty_sum/tr_total:.4f}",
                f"{val_em:.4f}", f"{val_ham:.4f}", f"{val_acc:.4f}"
            ])

        model_core = model.module if isinstance(model, nn.DataParallel) else model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            epochs_no_improve = 0
            torch.save(model_core.state_dict(), save_best_model_path)
            print(f" -> Best model saved at Ep {ep} (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1

        torch.save({
            'epoch': ep,
            'model_state': model_core.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'epochs_no_improve': epochs_no_improve
        }, save_checkpoint_path)

        trim_memory()
        gc.collect()

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping triggered at epoch {ep} (Patience: {cfg.patience})")
            break

    # Final Testing
    if os.path.exists(test_file):
        print(f"\n--- Running Evaluation on Test Set ---")
        test_dataset = NumericRAMDataset([test_file])
        test_sampler = TrainChunkBasedDynamicSampler(
            test_dataset,
            max_chunks=cfg.max_chunks_per_batch_eval,
            max_batch=cfg.max_batch_size_eval,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=eval_collator
        )

        eval_model = PartitionNetEq(window=cfg.window_size).to(device)
        weights_to_load = save_best_model_path if os.path.exists(save_best_model_path) else save_checkpoint_path
        eval_state = torch.load(weights_to_load, map_location=device)
        if isinstance(eval_state, dict) and 'model_state' in eval_state:
            eval_state = eval_state['model_state']
        eval_model.load_state_dict(clean_state_dict(eval_state))
        eval_model.eval()

        te_em, te_ham, te_acc, te_rep, test_logits, test_labels = evaluate(
            eval_model, test_loader, device, eval_collator
        )
        np.savez_compressed(save_preds_path, logits=test_logits, labels=test_labels)

        worst_diff, perm_agree, perm_acc_all = run_permutation_test(
            eval_model, test_dataset, collate_fn=eval_collator, device=device
        )

        print("=" * 70)
        print(f"Test Exact Match:     {te_em:.4f}")
        print(f"Test Hamming Loss:    {te_ham:.4f}")
        print(f"Test Topo Accuracy:   {te_acc*100:.2f}%")
        print(f"Perm Invariance Diff: {worst_diff:.3e}")
        print(f"Perm Topo Agreement:  {perm_agree*100:.2f}%")
        print(f"Perm Acc (All 24):    {perm_acc_all*100:.2f}%")
        print("=" * 70)

        results = {
            "test_exact_match": float(te_em),
            "test_hamming_loss": float(te_ham),
            "test_topo_acc": float(te_acc),
            "perm_max_diff": float(worst_diff),
            "perm_topo_agreement": float(perm_agree),
            "perm_topo_acc_all24": float(perm_acc_all),
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "classification_report": te_rep
        }
        with open(save_metrics_path, 'w') as f_out:
            json.dump(results, f_out, indent=4)
            print(f"Results saved to {save_metrics_path}")


if __name__ == "__main__":
    main()
