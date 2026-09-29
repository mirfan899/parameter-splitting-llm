#!/usr/bin/env python3
"""
Generate publication-quality empirical benchmark figures for DPS-LLM paper.
Generates two separate standalone figures specifically designed to fit into
single-column widths (\\columnwidth) in two-column conference formats like ACL.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

benchmarks = [
    'Qwen2.5-0.5B\n(GSM8K)',
    'SmolLM-360M\n(GSM8K)',
    'Qwen2.5-0.5B\n(Alpaca)',
    'SmolLM-360M\n(Alpaca)'
]

base_drops = [20.43, 1.03, 25.34, 0.71]
naive_drops = [13.83, 1.94, 24.03, 1.85]
dps_drops   = [14.09, 2.01, 24.12, 2.28]

c_base = '#6D28D9'    # Violet (validated categorical slot 1)
c_naive = '#D97706'   # Amber (validated categorical slot 2)
c_dps = '#2563EB'     # Blue (validated categorical slot 3)
c_margin = c_dps      # the margin belongs to DPS-LLM, so it wears its colour

# ==============================================================================
# Figure 1: Perplexity Improvement (% PPL Drop) - Single Column Optimized
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=300)

x = np.arange(len(benchmarks))
width = 0.26

rects1 = ax1.bar(x - width, base_drops, width, label='Base (Original)', color=c_base, edgecolor='#334155', linewidth=0.8, zorder=3)
rects2 = ax1.bar(x, naive_drops, width, label='Naive Split (Exact)', color=c_naive, edgecolor='#92400E', linewidth=0.8, zorder=3)
rects3 = ax1.bar(x + width, dps_drops, width, label='DPS-LLM (Ours)', color=c_dps, edgecolor='#1d4ed8', linewidth=0.8, zorder=3)

# Value labels on top of bars with clean tilt
for i in range(len(benchmarks)):
    ax1.annotate(f'{base_drops[i]:.2f}%',
                 xy=(rects1[i].get_x() + rects1[i].get_width() / 2, base_drops[i]),
                 xytext=(-1, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='medium', color='#334155',
                 rotation=20)
    ax1.annotate(f'{naive_drops[i]:.2f}%',
                 xy=(rects2[i].get_x() + rects2[i].get_width() / 2, naive_drops[i]),
                 xytext=(-1, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='medium', color='#334155',
                 rotation=20)
    ax1.annotate(f'{dps_drops[i]:.2f}%',
                 xy=(rects3[i].get_x() + rects3[i].get_width() / 2, dps_drops[i]),
                 xytext=(-1, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#334155',
                 rotation=20)

ax1.set_title('Validation Perplexity Improvement (% PPL Drop)', fontsize=11.5, fontweight='bold', pad=10, color='#0f172a')
ax1.set_ylabel('Perplexity Drop (%) [Higher is Better]', fontsize=10, fontweight='bold', color='#1e293b')
ax1.set_xticks(x)
ax1.set_xticklabels(benchmarks, fontsize=9.5, fontweight='medium')
ax1.set_ylim(0, 31)
ax1.grid(axis='y', zorder=0)
ax1.legend(loc='upper right', framealpha=0.92, edgecolor='#cbd5e1', fontsize=8.5)

# Alternating subtle background bands
ax1.axvspan(0.5, 1.5, color='#f8fafc', alpha=0.9, zorder=1)
ax1.axvspan(2.5, 3.5, color='#f8fafc', alpha=0.9, zorder=1)

fig1.tight_layout()
fig1_path_paper = '/home/iffi/Documents/Github/parameter-splitting-llm/paper/benchmark_ppl_comparison.png'
fig1_path_root  = '/home/iffi/Documents/Github/parameter-splitting-llm/benchmark_ppl_comparison.png'
fig1.savefig(fig1_path_paper, dpi=300, bbox_inches='tight')
fig1.savefig(fig1_path_root, dpi=300, bbox_inches='tight')
plt.close(fig1)

# ==============================================================================
# Figure 2: Symmetry-Breaking Margin (DPS-LLM vs. exact duplication) - single column
# ==============================================================================
INK, INK_MUTED, GRID = '#1E293B', '#64748B', '#E2E8F0'
dps_margins = [dps_drops[i] - naive_drops[i] for i in range(4)]
ratio_to_base = [d / b for d, b in zip(dps_drops, base_drops)]

fig2, ax2 = plt.subplots(figsize=(7.4, 4.3), dpi=300)
y_pos = np.arange(4)[::-1]                       # same order as the grouped bars, top to bottom
y_labels = [b.replace('\n', ' ') for b in benchmarks]
bars = ax2.barh(y_pos, dps_margins, height=0.5, color=c_dps, linewidth=0, zorder=3)
for bar, margin in zip(bars, dps_margins):
    ax2.annotate(f'+{margin:.2f} pp', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0), textcoords='offset points', ha='left', va='center',
                 fontsize=9.5, color=INK, zorder=4)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(y_labels, fontsize=9.5, color=INK)
ax2.set_xlim(0, 0.56)
ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_xlabel('Margin over naive split (percentage points of PPL reduction)', fontsize=9.5, color=INK)
ax2.set_title('DPS-LLM margin over exact duplication', fontsize=11, fontweight='bold', color=INK, loc='left', pad=10)
ax2.grid(axis='x', color=GRID, linestyle='-', linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
for side in ('top', 'right'):
    ax2.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax2.spines[side].set_color('#94A3B8'); ax2.spines[side].set_linewidth(0.8)
ax2.tick_params(colors=INK, labelsize=9, length=3, width=0.8)
ax2.tick_params(axis='y', length=0)
ax2.text(1.02, 1.0, 'DPS ÷ Base', transform=ax2.transAxes, ha='left', va='bottom',
         fontsize=9, color=INK_MUTED, fontweight='bold')
for y, rt in zip(y_pos, ratio_to_base):
    ax2.text(1.02, y, f'{rt:.2f}×', transform=ax2.get_yaxis_transform(), ha='left', va='center',
             fontsize=9.5, color=INK if rt > 1 else INK_MUTED, fontweight='bold' if rt > 1 else 'normal')

fig2.subplots_adjust(left=0.24, right=0.86, bottom=0.16, top=0.88)
fig2_path_paper = '/home/iffi/Documents/Github/parameter-splitting-llm/paper/benchmark_symmetry_margin.png'
fig2_path_root  = '/home/iffi/Documents/Github/parameter-splitting-llm/benchmark_symmetry_margin.png'
fig2.savefig(fig2_path_paper, dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig(fig2_path_root, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)

# ==============================================================================
# Figure 3: Composite two-panel figure used in the paper (figure*, full width)
#   (a) grouped bars of relative PPL reduction per condition
#   (b) horizontal bars of the DPS-LLM margin over exact duplication, with the
#       DPS / Base ratio as an aligned text column
# ==============================================================================
fig3, (axa, axb) = plt.subplots(1, 2, figsize=(14.0, 4.6), dpi=300,
                                gridspec_kw={'width_ratios': [1.35, 1.0], 'wspace': 0.28})

def clean_axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#94A3B8')
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9, length=3, width=0.8)

# ---- (a) grouped bars --------------------------------------------------------
xa = np.arange(len(benchmarks))
bw = 0.24
series = [('Base (no split)', base_drops, c_base),
          ('Naive split ($C=0$)', naive_drops, c_naive),
          ('DPS-LLM (ours)', dps_drops, c_dps)]
for k, (label, vals, colour) in enumerate(series):
    pos = xa + (k - 1) * (bw + 0.02)
    rects = axa.bar(pos, vals, bw, label=label, color=colour, linewidth=0, zorder=3)
    for r, v in zip(rects, vals):
        axa.annotate(f'{v:.2f}', xy=(r.get_x() + r.get_width() / 2, v), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=8, color=INK, zorder=4)
axa.set_xticks(xa)
axa.set_xticklabels([b.replace('\n', '\n') for b in benchmarks], fontsize=9.5, color=INK)
axa.set_ylabel('Relative PPL reduction (%)', fontsize=10, color=INK)
axa.set_ylim(0, 30)
axa.set_yticks([0, 10, 20, 30])
axa.grid(axis='y', color=GRID, linestyle='-', linewidth=0.8, zorder=0)
axa.set_axisbelow(True)
axa.legend(loc='upper left', frameon=False, fontsize=9, ncol=1, handlelength=1.2, handleheight=0.9)
axa.set_title('(a) Validation perplexity reduction after fine-tuning', fontsize=11, fontweight='bold', color=INK, loc='left', pad=10)
clean_axes(axa)

# ---- (b) margin over naive duplication ---------------------------------------
yb = np.arange(len(benchmarks))[::-1]           # same order as (a), top to bottom
labels_b = [b.replace('\n', ' ') for b in benchmarks]
bars_b = axb.barh(yb, dps_margins, height=0.5, color=c_dps, linewidth=0, zorder=3)
for r, m in zip(bars_b, dps_margins):
    axb.annotate(f'+{m:.2f} pp', xy=(r.get_width(), r.get_y() + r.get_height() / 2), xytext=(5, 0),
                 textcoords='offset points', ha='left', va='center', fontsize=9, color=INK, zorder=4)
axb.set_yticks(yb)
axb.set_yticklabels(labels_b, fontsize=9.5, color=INK)
axb.set_xlim(0, 0.56)
axb.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axb.set_xlabel('Margin over naive split (percentage points of PPL reduction)', fontsize=9.5, color=INK)
axb.grid(axis='x', color=GRID, linestyle='-', linewidth=0.8, zorder=0)
axb.set_axisbelow(True)
axb.set_title('(b) DPS-LLM margin over exact duplication', fontsize=11, fontweight='bold', color=INK, loc='left', pad=10)
clean_axes(axb)
axb.tick_params(axis='y', length=0)
# aligned text column: ratio of DPS-LLM reduction to the base model's reduction
axb.text(1.02, 1.0, 'DPS ÷ Base', transform=axb.transAxes, ha='left', va='bottom',
         fontsize=9, color=INK_MUTED, fontweight='bold')
for y, rt in zip(yb, ratio_to_base):
    axb.text(1.02, y, f'{rt:.2f}×', transform=axb.get_yaxis_transform(), ha='left', va='center',
             fontsize=9.5, color=INK if rt > 1 else INK_MUTED,
             fontweight='bold' if rt > 1 else 'normal')

fig3.subplots_adjust(left=0.06, right=0.94, bottom=0.17, top=0.88)
fig3_path_paper = '/home/iffi/Documents/Github/parameter-splitting-llm/paper/multi_model_benchmark_comparison.png'
fig3_path_root  = '/home/iffi/Documents/Github/parameter-splitting-llm/multi_model_benchmark_comparison.png'
fig3.savefig(fig3_path_paper, dpi=300, bbox_inches='tight', facecolor='white')
fig3.savefig(fig3_path_root, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f" 3. {fig3_path_paper}")
