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

c_base = '#64748B'    # Slate Gray
c_naive = '#F59E0B'   # Amber / Orange
c_dps = '#2563EB'     # Deep Royal Blue
c_margin = '#059669'  # Emerald Green

# ==============================================================================
# Figure 1: Perplexity Improvement (% PPL Drop) - Single Column Optimized
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=300)

x = np.arange(len(benchmarks))
width = 0.26

rects1 = ax1.bar(x - width, base_drops, width, label='Base (Original)', color=c_base, edgecolor='#334155', linewidth=0.8, zorder=3)
rects2 = ax1.bar(x, naive_drops, width, label='Naive Split (Exact)', color=c_naive, edgecolor='#b45309', linewidth=0.8, zorder=3)
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
                 ha='center', va='bottom', fontsize=8.5, fontweight='medium', color='#b45309',
                 rotation=20)
    ax1.annotate(f'{dps_drops[i]:.2f}%',
                 xy=(rects3[i].get_x() + rects3[i].get_width() / 2, dps_drops[i]),
                 xytext=(-1, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1d4ed8',
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
# Figure 2: Symmetry-Breaking Advantage & Capacity Relief - Single Column Optimized
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(7.4, 4.5), dpi=300)

y_pos = np.arange(4)
y_labels = [
    'Qwen2.5 (GSM8K)',
    'SmolLM (GSM8K)',
    'Qwen2.5 (Alpaca)',
    'SmolLM (Alpaca)'
]

dps_margins = [dps_drops[i] - naive_drops[i] for i in range(4)]

bars = ax2.barh(y_pos, dps_margins, height=0.48, color=c_margin, edgecolor='#047857', linewidth=1.0, zorder=3)

for bar, margin in zip(bars, dps_margins):
    w = bar.get_width()
    ax2.annotate(f'+{margin:.2f}% gain',
                 xy=(w, bar.get_y() + bar.get_height() / 2),
                 xytext=(7, 0),
                 textcoords="offset points",
                 ha='left', va='center', fontsize=9.5, fontweight='bold', color='#065f46')

ax2.set_title('Symmetry-Breaking Margin (DPS-LLM vs. Naive)', fontsize=11.5, fontweight='bold', pad=10, color='#0f172a')
ax2.set_xlabel('Additional PPL Drop over Exact Duplication (%)\n[Calculated Noise vs. Naive Copy]', fontsize=9.5, fontweight='bold', color='#1e293b')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(y_labels, fontsize=9.5, fontweight='medium')
ax2.set_xlim(0, 0.68)
ax2.grid(axis='x', zorder=0)

# Callout boxes positioned cleanly on the right half
ax2.text(0.50, 0.22, '100% Win Rate (4/4)\nAnti-symmetric noise\nbreaks degeneracy',
         transform=ax2.transAxes, fontsize=8.8, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5', edgecolor='#10b981', alpha=0.92),
         ha='left', va='center', color='#065f46')

ax2.text(0.50, 0.72, 'SmolLM Capacity Relief:\n• Alpaca: 3.21× gain\n• GSM8K: 1.95× gain',
         transform=ax2.transAxes, fontsize=8.8, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eff6ff', edgecolor='#3b82f6', alpha=0.92),
         ha='left', va='center', color='#1e40af')

fig2.tight_layout()
fig2_path_paper = '/home/iffi/Documents/Github/parameter-splitting-llm/paper/benchmark_symmetry_margin.png'
fig2_path_root  = '/home/iffi/Documents/Github/parameter-splitting-llm/benchmark_symmetry_margin.png'
fig2.savefig(fig2_path_paper, dpi=300, bbox_inches='tight')
fig2.savefig(fig2_path_root, dpi=300, bbox_inches='tight')
plt.close(fig2)

print("Generated separate figures successfully:")
print(f" 1. {fig1_path_paper}")
print(f" 2. {fig2_path_paper}")
