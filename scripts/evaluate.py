import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from partitionnet.config import TrainingConfig
from partitionnet.models.partitionnet import PartitionNetEq
from partitionnet.data.dataset import NumericRAMDataset
from partitionnet.data.sampler import TrainChunkBasedDynamicSampler
from partitionnet.data.collate import ChunkCollator
from partitionnet.utils.memory import clean_state_dict
from partitionnet.utils.metrics import compute_metrics
from partitionnet.utils.invariance import run_permutation_test


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PartitionNet checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test HDF5 file")
    parser.add_argument("--output-json", type=str, default="evaluation_results.json", help="Path to output JSON")
    parser.add_argument("--output-preds", type=str, default="test_preds.npz", help="Path to save predictions NPZ")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--perm-test", action="store_true", help="Run permutation invariance test across 24 orders")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = TrainingConfig()

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = PartitionNetEq().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(clean_state_dict(state))
    model.eval()

    print(f"Loading test dataset: {args.test_file}")
    dataset = NumericRAMDataset([args.test_file])
    sampler = TrainChunkBasedDynamicSampler(
        dataset,
        max_chunks=cfg.max_chunks_per_batch_eval,
        max_batch=cfg.max_batch_size_eval,
        shuffle=False
    )
    collator = ChunkCollator(
        max_batch_size=cfg.max_batch_size_eval,
        max_chunks_limit=cfg.max_chunks_limit,
        window_size=cfg.window_size
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

    total_samples = len(dataset)
    logits_arr = np.empty((total_samples, 4), dtype=np.float32)
    labels_arr = np.empty((total_samples, 4), dtype=np.float32)
    ptr = 0

    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in loader:
            bs = labels.size(0)
            inputs = inputs.to(device, non_blocking=True)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(inputs)
            logits_arr[ptr: ptr + bs] = out.cpu().float().numpy()
            labels_arr[ptr: ptr + bs] = labels.numpy()
            ptr += bs

    em, ham, acc, rep = compute_metrics(logits_arr, labels_arr)
    np.savez_compressed(args.output_preds, logits=logits_arr, labels=labels_arr)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Exact Match:   {em:.4f}")
    print(f"Hamming Loss:  {ham:.4f}")
    print(f"Topo Accuracy: {acc*100:.2f}%")
    print("=" * 60)

    results = {
        "exact_match": float(em),
        "hamming_loss": float(ham),
        "topology_accuracy": float(acc),
        "classification_report": rep
    }

    if args.perm_test:
        print("\nRunning Permutation Invariance Test (24 Taxon Orders)...")
        worst_diff, agree, perm_acc = run_permutation_test(
            model, dataset, collate_fn=collator, device=device
        )
        print(f"Max Logit Diff:    {worst_diff:.3e}")
        print(f"Topology Agreement:{agree*100:.2f}%")
        print(f"Perm Acc (All 24): {perm_acc*100:.2f}%")
        results["permutation_worst_diff"] = float(worst_diff)
        results["permutation_topology_agreement"] = float(agree)
        results["permutation_topo_accuracy_all24"] = float(perm_acc)

    with open(args.output_json, 'w') as f_out:
        json.dump(results, f_out, indent=4)
        print(f"\nExported metrics to {args.output_json}")


if __name__ == "__main__":
    main()
