"""Generate publication-quality charts for MedusaBitNet results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

C = {
    'blue': '#2563EB', 'red': '#DC2626', 'green': '#16A34A',
    'purple': '#9333EA', 'orange': '#EA580C', 'gray': '#6B7280',
    'dark': '#1F2937', 'gold': '#D97706',
}

# ============================================================================
# 1. Training loss curve
# ============================================================================
steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
         160, 170, 180, 280, 440, 590, 740, 890, 1050, 1200, 1360, 1510,
         1660, 1820, 1990]
losses = [9.85, 7.73, 6.42, 5.64, 5.22, 4.90, 4.76, 4.66, 4.55, 4.53,
          4.49, 4.46, 4.41, 4.33, 4.32, 4.28, 4.28, 4.24, 4.12, 4.02,
          3.91, 3.85, 3.73, 3.63, 3.54, 3.50, 3.42, 3.35, 3.34, 3.32]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, losses, color=C['blue'], linewidth=2.5, marker='o', markersize=3)
ax.set_xlabel('Training Step')
ax.set_ylabel('Medusa Loss')
ax.set_title('MedusaBitNet Training Loss\nBitNet b1.58 2B-4T + 4 Medusa Heads')
ax.set_xlim(0, 2050)
ax.set_ylim(3, 10.5)
ax.grid(True, alpha=0.3)
ax.axhline(y=3.32, color=C['gray'], linestyle='--', alpha=0.5, label='Final: 3.32')
ax.legend(loc='upper right')
ax.annotate('Loss 9.85', xy=(10, 9.85), xytext=(100, 9.5),
            arrowprops=dict(arrowstyle='->', color=C['gray']),
            fontsize=10, color=C['gray'])
ax.annotate('Loss 3.32', xy=(1990, 3.32), xytext=(1700, 4.0),
            arrowprops=dict(arrowstyle='->', color=C['blue']),
            fontsize=10, color=C['blue'], fontweight='bold')
fig.savefig(f'{OUT}/training_loss.png')
plt.close()
print(f"Saved {OUT}/training_loss.png")

# ============================================================================
# 2. Head accuracy curves
# ============================================================================
acc_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
             150, 160, 280, 440, 590, 740, 890, 1050, 1200, 1360, 1510,
             1660, 1820, 1990]
acc1 = [0.672, 0.617, 0.574, 0.637, 0.656, 0.613, 0.664, 0.602, 0.555,
        0.613, 0.605, 0.559, 0.602, 0.617, 0.594, 0.617, 0.688, 0.617,
        0.609, 0.625, 0.652, 0.641, 0.664, 0.664, 0.625, 0.676, 0.664, 0.641]
acc2 = [0.031, 0.125, 0.125, 0.234, 0.281, 0.309, 0.266, 0.277, 0.344,
        0.266, 0.305, 0.258, 0.305, 0.297, 0.285, 0.312, 0.340, 0.336,
        0.328, 0.320, 0.406, 0.391, 0.359, 0.441, 0.316, 0.426, 0.340, 0.375]
acc3 = [0.023, 0.059, 0.070, 0.121, 0.121, 0.129, 0.188, 0.121, 0.164,
        0.141, 0.195, 0.137, 0.191, 0.172, 0.148, 0.109, 0.207, 0.188,
        0.219, 0.203, 0.152, 0.234, 0.227, 0.180, 0.203, 0.238, 0.188, 0.191]
acc4 = [0.016, 0.043, 0.070, 0.074, 0.117, 0.113, 0.102, 0.102, 0.074,
        0.082, 0.098, 0.082, 0.121, 0.129, 0.141, 0.133, 0.121, 0.133,
        0.145, 0.113, 0.105, 0.109, 0.148, 0.109, 0.219, 0.160, 0.148, 0.176]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(acc_steps, acc1, color=C['blue'], linewidth=2, label='Head 1 (next token)', marker='o', markersize=2)
ax.plot(acc_steps, acc2, color=C['red'], linewidth=2, label='Head 2 (t+2)', marker='s', markersize=2)
ax.plot(acc_steps, acc3, color=C['green'], linewidth=2, label='Head 3 (t+3)', marker='^', markersize=2)
ax.plot(acc_steps, acc4, color=C['purple'], linewidth=2, label='Head 4 (t+4)', marker='D', markersize=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Top-1 Accuracy')
ax.set_title('Medusa Head Accuracy During Training\n4 Speculative Heads on BitNet b1.58 2B-4T')
ax.set_xlim(0, 2050)
ax.set_ylim(0, 0.8)
ax.legend(loc='center right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.65, color=C['blue'], linestyle=':', alpha=0.3)
ax.text(1800, 0.66, 'backbone baseline', fontsize=9, color=C['blue'], alpha=0.6)
fig.savefig(f'{OUT}/head_accuracy.png')
plt.close()
print(f"Saved {OUT}/head_accuracy.png")

# ============================================================================
# 3. Medusa vs Vanilla speedup (THE KEY CHART)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: acceptance rates
ax = axes[0]
heads = ['Head 1\n(t+1)', 'Head 2\n(t+2)', 'Head 3\n(t+3)', 'Head 4\n(t+4)']
rates = [0.676, 0.332, 0.142, 0.063]
colors = [C['blue'], C['red'], C['green'], C['purple']]
bars = ax.bar(heads, rates, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{val*100:.1f}%', ha='center', fontsize=13, fontweight='bold', color=C['dark'])
ax.set_ylabel('Acceptance Rate')
ax.set_title('Medusa Speculative Acceptance Rates\n(greedy verification, 40K positions)')
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)

# Right: speedup comparison
ax = axes[1]
methods = ['Vanilla\nBitNet', 'MedusaBitNet\n(4 heads)']
tokens_per_step = [1.0, 2.21]
bar_colors = [C['gray'], C['gold']]
bars = ax.bar(methods, tokens_per_step, color=bar_colors, width=0.45, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, tokens_per_step):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontsize=16, fontweight='bold', color=C['dark'])
ax.set_ylabel('Tokens per Backbone Step')
ax.set_title('Effective Throughput: 2.21x Speedup\nFirst Medusa + BitNet Integration')
ax.set_ylim(0, 3.0)
ax.grid(axis='y', alpha=0.3)
# Add speedup arrow
ax.annotate('2.21x', xy=(1, 2.21), xytext=(0.5, 2.6),
            fontsize=18, fontweight='bold', color=C['gold'],
            arrowprops=dict(arrowstyle='->', lw=2, color=C['gold']),
            ha='center')

plt.tight_layout()
fig.savefig(f'{OUT}/medusa_speedup.png')
plt.close()
print(f"Saved {OUT}/medusa_speedup.png")

# ============================================================================
# 4. Architecture diagram
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('MedusaBitNet: Speculative Decoding with Ternary Weights\n2.21x throughput, 1.7% model size overhead', fontsize=16, fontweight='bold', pad=20)

backbone = mpatches.FancyBboxPatch((0.5, 2), 4, 3.5, boxstyle="round,pad=0.15",
                                     facecolor='#DBEAFE', edgecolor=C['blue'], linewidth=2)
ax.add_patch(backbone)
ax.text(2.5, 5.0, 'BitNet b1.58 2B-4T', ha='center', fontsize=13, fontweight='bold', color=C['dark'])
ax.text(2.5, 4.4, 'Frozen Backbone', ha='center', fontsize=11, color=C['gray'])
ax.text(2.5, 3.8, '30 layers, 2560 hidden', ha='center', fontsize=10, color=C['gray'])
ax.text(2.5, 3.2, 'Ternary weights {-1, 0, 1}', ha='center', fontsize=10, color=C['gray'])
ax.text(2.5, 2.6, '751 MB (I2_S quantized)', ha='center', fontsize=10, fontweight='bold', color=C['blue'])

ax.annotate('', xy=(5.5, 4), xytext=(4.7, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color=C['dark']))
ax.text(5.1, 4.3, 'hidden\nstates', ha='center', fontsize=9, color=C['dark'])

for i, (y, label, acc, color) in enumerate([
    (5.5, 'Head 1: t+1', '67.6%', C['blue']),
    (4.5, 'Head 2: t+2', '33.2%', C['red']),
    (3.5, 'Head 3: t+3', '14.2%', C['green']),
    (2.5, 'Head 4: t+4', '6.3%', C['purple']),
]):
    head = mpatches.FancyBboxPatch((5.8, y-0.35), 2.8, 0.7, boxstyle="round,pad=0.1",
                                     facecolor='#FEF3C7', edgecolor=color, linewidth=1.5)
    ax.add_patch(head)
    ax.text(7.2, y, f'{label} ({acc})', ha='center', va='center', fontsize=10, fontweight='bold', color=color)

ax.annotate('', xy=(9.5, 4), xytext=(8.8, 4),
            arrowprops=dict(arrowstyle='->', lw=2, color=C['dark']))

spec = mpatches.FancyBboxPatch((9.5, 2.5), 2, 3, boxstyle="round,pad=0.15",
                                 facecolor='#DCFCE7', edgecolor=C['green'], linewidth=2)
ax.add_patch(spec)
ax.text(10.5, 4.8, '2.21x', ha='center', fontsize=20, fontweight='bold', color=C['gold'])
ax.text(10.5, 4.1, 'Speedup', ha='center', fontsize=14, fontweight='bold', color=C['dark'])
ax.text(10.5, 3.4, '2.21 tokens', ha='center', fontsize=11, color=C['gray'])
ax.text(10.5, 2.9, 'per step', ha='center', fontsize=11, color=C['gray'])

ax.text(2.5, 1.3, 'Input Tokens', ha='center', fontsize=11, color=C['dark'])
ax.annotate('', xy=(2.5, 1.8), xytext=(2.5, 1.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=C['dark']))

ax.text(6, 0.5, 'AMD Strix Halo  |  Ryzen AI MAX+ 395  |  Zen 5 + RDNA 3.5  |  93GB unified RAM',
        ha='center', fontsize=10, color=C['gray'], style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6', edgecolor='#D1D5DB'))

fig.savefig(f'{OUT}/architecture.png')
plt.close()
print(f"Saved {OUT}/architecture.png")

# ============================================================================
# 5. Performance summary
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: throughput with Medusa speedup
ax = axes[0]
categories = ['Vanilla\n(75 tok/s)', 'MedusaBitNet\n(est. 166 tok/s)']
throughput = [75.4, 75.4 * 2.21]
bar_colors = [C['gray'], C['gold']]
bars = ax.bar(categories, throughput, color=bar_colors, width=0.45, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.0f}', ha='center', fontsize=14, fontweight='bold', color=C['dark'])
ax.set_ylabel('Effective Tokens / Second')
ax.set_title('Estimated Throughput on Strix Halo\n(Zen 5, 16 threads, I2_S quantized)')
ax.set_ylim(0, 200)
ax.grid(axis='y', alpha=0.3)

# Right: model sizes
ax = axes[1]
models = ['Backbone\n(I2_S)', '+ Medusa\nHeads (f16)', 'Total\nMedusaBitNet']
sizes = [751, 13, 764]
colors_bar = [C['blue'], C['orange'], C['green']]
bars = ax.bar(models, sizes, color=colors_bar, width=0.5, edgecolor='white')
for bar, val in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val} MB', ha='center', fontsize=12, fontweight='bold', color=C['dark'])
ax.set_ylabel('Size (MB)')
ax.set_title('Model Size Breakdown\nMedusa Heads Add Only 1.7% Overhead')
ax.set_ylim(0, 900)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(f'{OUT}/performance_summary.png')
plt.close()
print(f"Saved {OUT}/performance_summary.png")

# ============================================================================
# 6. Pipeline timeline
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 3))

stages = [
    ('Tokenize', 0, 0.5, C['green']),
    ('Cache Hidden\nStates (CPU)', 0.5, 4.6, C['blue']),
    ('Train Medusa\nHeads (CPU)', 5.1, 6.8, C['red']),
    ('Convert\nGGUF', 11.9, 0.5, C['purple']),
]

for label, start, duration, color in stages:
    ax.barh(0, duration, left=start, height=0.6, color=color, alpha=0.85, edgecolor='white', linewidth=1)
    ax.text(start + duration/2, 0, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax.set_xlim(-0.2, 13)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Hours')
ax.set_title('MedusaBitNet Full Pipeline Timeline (AMD Strix Halo)', fontsize=13, fontweight='bold')
ax.set_yticks([])
ax.grid(axis='x', alpha=0.3)

ax.text(0.25, -0.4, '~30s', ha='center', fontsize=8, color=C['gray'])
ax.text(2.8, -0.4, '~4.1 hours', ha='center', fontsize=8, color=C['gray'])
ax.text(8.5, -0.4, '~6.8 hours', ha='center', fontsize=8, color=C['gray'])
ax.text(12.15, -0.4, '~1 min', ha='center', fontsize=8, color=C['gray'])

fig.savefig(f'{OUT}/pipeline_timeline.png')
plt.close()
print(f"Saved {OUT}/pipeline_timeline.png")

print("\nAll charts generated!")
