"""
Generate figures for Assignment 4 paper from training and evaluation results.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="whitegrid", font_scale=1.1)
outdir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(outdir, exist_ok=True)

# =========================================================================
# Training data from 7 completed epochs
# =========================================================================
epochs = np.arange(1, 8)
train_loss   = [1.1295, 0.2883, 0.1569, 0.1279, 0.1073, 0.1026, 0.0962]
train_proto  = [0.8810, 0.9802, 0.9873, 0.9888, 0.9908, 0.9911, 0.9917]
dev_loss     = [0.8702, 0.4220, 0.4383, 0.6007, 0.6890, 0.3097, 0.5662]
dev_proto    = [0.9250, 0.9420, 0.9465, 0.9598, 0.9510, 0.9685, 0.9645]
epoch_time_s = [2660.1, 2657.8, 2651.6, 2668.7, 2646.4, 2653.8, 2649.9]


# -------------------------------------------------------------------------
# Figure 1: Training & Dev Loss over Epochs
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, train_loss, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Training Loss')
ax.plot(epochs, dev_loss, 's--', color='#F44336', linewidth=2, markersize=7, label='Dev Loss')
# Mark best dev checkpoints
best_epochs = [1, 2, 6]
for e in best_epochs:
    ax.annotate('best', xy=(e, dev_loss[e-1]), xytext=(e+0.3, dev_loss[e-1]+0.08),
                fontsize=8, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Dev Loss Over Epochs')
ax.legend()
ax.set_xticks(epochs)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_loss_curves.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_loss_curves.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Prototypical Accuracy over Epochs
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, [x*100 for x in train_proto], 'o-', color='#4CAF50', linewidth=2, markersize=7, label='Train Proto Acc')
ax.plot(epochs, [x*100 for x in dev_proto], 's--', color='#FF9800', linewidth=2, markersize=7, label='Dev Proto Acc')
ax.set_xlabel('Epoch')
ax.set_ylabel('Prototypical Accuracy (%)')
ax.set_title('Prototypical Classification Accuracy Over Epochs')
ax.legend(loc='lower right')
ax.set_xticks(epochs)
ax.set_ylim([85, 100])
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_proto_accuracy.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_proto_accuracy.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Dual-Stream Feature Architecture (bar chart of dimensions)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
components = ['WavLM\n(SSL)', 'HuBERT\n(SSL)', 'MFCC\n(Cepstral)', 'LFCC\n(Cepstral)', 'CQCC\n(Cepstral)']
dims = [768, 768, 40, 40, 40]
colors = ['#1976D2', '#1565C0', '#E65100', '#EF6C00', '#F57C00']
bars = ax.bar(components, dims, color=colors, edgecolor='white', linewidth=1.5)
for bar, d in zip(bars, dims):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(d), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Feature Dimensions')
ax.set_title('Dual-Stream Feature Composition (Total: 1,656-dim)')
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_feature_dims.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_feature_dims.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Train vs Dev Performance Comparison (grouped bar)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
# Use best epoch (epoch 6) metrics
best_idx = 5  # epoch 6 (0-indexed)
metrics_names = ['Loss', 'Proto Acc']
train_vals = [train_loss[best_idx], train_proto[best_idx]]
dev_vals = [dev_loss[best_idx], dev_proto[best_idx]]

x = np.arange(len(metrics_names))
width = 0.30
bars1 = ax.bar(x - width/2, train_vals, width, label='Training', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + width/2, dev_vals, width, label='Dev', color='#F44336', edgecolor='white')
ax.set_ylabel('Value')
ax.set_title('Best Epoch (Epoch 6) — Training vs Dev Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_best_epoch_comparison.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_best_epoch_comparison.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 5: Training Time per Epoch
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(epochs, [t/60 for t in epoch_time_s], color='#7B1FA2', edgecolor='white', width=0.6)
ax.axhline(y=np.mean(epoch_time_s)/60, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(epoch_time_s)/60:.1f} min')
ax.set_xlabel('Epoch')
ax.set_ylabel('Time (minutes)')
ax.set_title('Training Time per Epoch (1,000 episodes/epoch)')
ax.set_xticks(epochs)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_training_time.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_training_time.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 6: Convergence Analysis — Loss Reduction Rate
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
loss_reduction = [0] + [train_loss[i-1] - train_loss[i] for i in range(1, len(train_loss))]
ax.bar(epochs, loss_reduction, color='#00897B', edgecolor='white', width=0.6)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss Reduction from Previous Epoch')
ax.set_title('Training Loss Reduction Per Epoch (Convergence Rate)')
ax.set_xticks(epochs)
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_convergence_rate.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_convergence_rate.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 7: Dataset Composition (pie chart)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
# ASVspoof 2019 LA train partition: 2,580 bonafide + 22,800 spoof (6 attacks)
labels_pie = ['Bonafide\n(genuine)', 'A01 (TTS)', 'A02 (TTS)', 'A03 (TTS)',
              'A04 (VC)', 'A05 (VC)', 'A06 (VC)']
# Approximate per-attack distribution for train set (25,380 total)
bonafide_count = 2580
spoof_per_attack = (25380 - 2580) // 6  # ~3800 each
sizes = [bonafide_count] + [spoof_per_attack]*6
colors_pie = ['#4CAF50'] + sns.color_palette("Reds", 6).as_hex()
explode = [0.05] + [0]*6
ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
       autopct='%1.1f%%', startangle=140, textprops={'fontsize': 9})
ax.set_title('ASVspoof 2019 LA Training Set Composition\n(25,380 samples)')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig_dataset_composition.pdf'), dpi=300)
plt.savefig(os.path.join(outdir, 'fig_dataset_composition.png'), dpi=300)
plt.close()

print("All figures generated in:", outdir)
print("Files:")
for f in sorted(os.listdir(outdir)):
    print(f"  {f}")


# =========================================================================
# Evaluation figures from few-shot results
# =========================================================================
eval_results_path = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "evaluation_results.json",
)

if os.path.exists(eval_results_path):
    with open(eval_results_path, "r") as f:
        eval_results = json.load(f)

    attacks = sorted(eval_results.keys())
    shot_labels = sorted(
        next(iter(eval_results.values())).keys(),
        key=lambda label: int(label.split("-")[0]),
    )
    shot_values = [int(label.split("-")[0]) for label in shot_labels]

    eer_matrix = np.array([
        [eval_results[attack][shot]["eer"] * 100 for shot in shot_labels]
        for attack in attacks
    ])
    acc_matrix = np.array([
        [eval_results[attack][shot]["accuracy"] * 100 for shot in shot_labels]
        for attack in attacks
    ])

    # ---------------------------------------------------------------------
    # Figure 8: EER by attack and shot count
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    shot_colors = ['#1E88E5', '#FB8C00', '#43A047']
    shot_markers = ['o', 's', '^']
    for idx, shot in enumerate(shot_labels):
        ax.plot(
            attacks,
            eer_matrix[:, idx],
            marker=shot_markers[idx],
            linewidth=2,
            markersize=6,
            color=shot_colors[idx],
            label=shot,
        )
    ax.set_xlabel('Held-Out Attack Type')
    ax.set_ylabel('Equal Error Rate (%)')
    ax.set_title('Few-Shot Evaluation EER Across Held-Out Attacks')
    ax.legend(title='Support Size')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_eval_eer_by_attack.pdf'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_eval_eer_by_attack.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # Figure 9: Accuracy heatmap by attack and shot count
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.heatmap(
        acc_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Accuracy (%)'},
        xticklabels=[str(v) for v in shot_values],
        yticklabels=attacks,
        ax=ax,
    )
    ax.set_xlabel('Support Examples per Class')
    ax.set_ylabel('Held-Out Attack Type')
    ax.set_title('Few-Shot Evaluation Accuracy Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_eval_accuracy_heatmap.pdf'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_eval_accuracy_heatmap.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # Figure 10: Best EER achieved for each attack
    # ---------------------------------------------------------------------
    best_shot_indices = np.argmin(eer_matrix, axis=1)
    best_eer = eer_matrix[np.arange(len(attacks)), best_shot_indices]
    best_shots = [shot_values[idx] for idx in best_shot_indices]
    ranking = np.argsort(best_eer)[::-1]
    ranked_attacks = [attacks[idx] for idx in ranking]
    ranked_eer = best_eer[ranking]
    ranked_best_shots = [best_shots[idx] for idx in ranking]
    bar_colors = [shot_colors[shot_values.index(shot)] for shot in ranked_best_shots]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(ranked_attacks, ranked_eer, color=bar_colors, edgecolor='white')
    for bar, shot in zip(bars, ranked_best_shots):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f'{shot}-shot',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )
    ax.set_xlabel('Held-Out Attack Type')
    ax.set_ylabel('Best Equal Error Rate (%)')
    ax.set_title('Best Few-Shot EER Achieved for Each Attack')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_eval_best_eer_ranked.pdf'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_eval_best_eer_ranked.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # Figure 11: Average performance versus shot count
    # ---------------------------------------------------------------------
    avg_eer = eer_matrix.mean(axis=0)
    avg_acc = acc_matrix.mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    line1 = ax1.plot(
        shot_values,
        avg_eer,
        marker='o',
        linewidth=2,
        color='#D81B60',
        label='Average EER',
    )
    ax1.set_xlabel('Support Examples per Class')
    ax1.set_ylabel('Average EER (%)', color='#D81B60')
    ax1.tick_params(axis='y', labelcolor='#D81B60')
    ax1.set_xticks(shot_values)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        shot_values,
        avg_acc,
        marker='s',
        linewidth=2,
        color='#3949AB',
        label='Average Accuracy',
    )
    ax2.set_ylabel('Average Accuracy (%)', color='#3949AB')
    ax2.tick_params(axis='y', labelcolor='#3949AB')

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.set_title('Average Evaluation Performance Versus Shot Count')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_eval_average_performance.pdf'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_eval_average_performance.png'), dpi=300)
    plt.close()
