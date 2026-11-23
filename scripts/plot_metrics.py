import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def bipolar_to_topo(bit_arr):
    """Maps 4-bit bipolar vector to quartet topology class index (1, 2, or 3)."""
    bit_str = "".join(['1' if x > 0 else '0' for x in bit_arr])
    mapping = {
        "1100": 1, "0011": 1,
        "1010": 2, "0101": 2,
        "1001": 3, "0110": 3
    }
    return mapping.get(bit_str, -1)


def decode_topos(logits: np.ndarray) -> np.ndarray:
    """Decodes array of raw 4-dim logits into quartet topology labels."""
    top2_idx = np.argsort(-logits, axis=1)[:, :2]
    preds_top2 = np.full_like(logits, -1.0)
    np.put_along_axis(preds_top2, top2_idx, 1.0, axis=1)
    return np.array([bipolar_to_topo(b) for b in preds_top2])


def plot_training_curves(log_path: str, output_path: str):
    epochs, tr_loss, tr_margin, tr_penalty = [], [], [], []
    va_exact, va_ham, va_acc = [], [], []

    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            tr_loss.append(float(row['Tr_Loss']))
            tr_margin.append(float(row['Tr_Margin']))
            tr_penalty.append(float(row['Tr_Penalty']))
            va_exact.append(float(row['Va_Exact']) * 100)
            va_ham.append(float(row['Va_Ham']))
            va_acc.append(float(row['Va_TopoAcc']) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Loss curves
    ax1.plot(epochs, tr_loss, label='Total Loss', color='#1f77b4', lw=2)
    ax1.plot(epochs, tr_margin, label='Symmetric Margin Loss', color='#ff7f0e', lw=1.8, linestyle='--')
    ax1.plot(epochs, tr_penalty, label='Constraint Penalty', color='#2ca02c', lw=1.8, linestyle=':')
    ax1.set_title('Training Loss Decomposition (PartitionNet V28)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss Value', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, fontsize=10)

    # Subplot 2: Validation Accuracy & Exact Match
    ax2.plot(epochs, va_acc, label='Val Topology Accuracy (%)', color='#d62728', lw=2)
    ax2.plot(epochs, va_exact, label='Val Exact Match (%)', color='#9467bd', lw=1.8, linestyle='--')
    
    best_idx = np.argmax(va_acc)
    best_ep = epochs[best_idx]
    best_acc_val = va_acc[best_idx]
    ax2.scatter([best_ep], [best_acc_val], color='red', s=80, zorder=5)
    ax2.annotate(f'Best: {best_acc_val:.2f}%\n(Epoch {best_ep})',
                 xy=(best_ep, best_acc_val),
                 xytext=(best_ep - 5, best_acc_val - 2.5),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontweight='bold', fontsize=9)

    ax2.set_title('Validation Performance Over Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {output_path}")


def plot_confusion_matrix(preds_path: str, output_path: str):
    data = np.load(preds_path)
    logits = data['logits']
    labels = data['labels']

    true_topos = np.array([bipolar_to_topo(b) for b in labels])
    pred_topos = decode_topos(logits)

    classes = [1, 2, 3]
    class_names = ['Topology 1\n(1,2)|(3,4)', 'Topology 2\n(1,3)|(2,4)', 'Topology 3\n(1,4)|(2,3)']
    cm = confusion_matrix(true_topos, pred_topos, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=class_names, yticklabels=class_names,
           title='Quartet Topology Confusion Matrix\n(Test Set: 51,241 Samples)',
           ylabel='True Topology',
           xlabel='Predicted Topology')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, val_str,
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')

    log_path = os.path.join(results_dir, 'training_log_eq.csv')
    preds_path = os.path.join(results_dir, 'pnet_v28_eq_preds.npz')

    curves_out = os.path.join(results_dir, 'training_curves.png')
    cm_out = os.path.join(results_dir, 'confusion_matrix.png')

    if os.path.exists(log_path):
        plot_training_curves(log_path, curves_out)
    else:
        print(f"File not found: {log_path}")

    if os.path.exists(preds_path):
        plot_confusion_matrix(preds_path, cm_out)
    else:
        print(f"File not found: {preds_path}")


if __name__ == '__main__':
    main()
