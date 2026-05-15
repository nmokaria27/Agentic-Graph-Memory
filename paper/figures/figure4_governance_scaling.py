"""
Figure 4: Governance Scales While Extraction Drifts
====================================================
The key claim: at 100 documents, the governance layer (routing, completeness,
audit, domain coverage) remains at or above 0.98 even as extraction-side
metrics degrade.

Two-panel design:
  Left  — extraction-side metrics across 3 scales: 10, 50, 100 docs
            · Entity F1 (stays roughly flat, then slightly drifts)
            · Triple F1 (starts high under governance at 10 docs, then drifts)
            · Triple hallucination rate (right y-axis, lower is better)
  Right — governance-side metrics at 100 docs shown as a horizontal
            "scorecard" bar chart, all sitting at or above 0.98.

Data source: paper/RESULTS.md, Tables 1a and 2.

Run:
    python paper/figures/figure4_governance_scaling.py
Outputs:
    paper/figures/figure4_governance_scaling.pdf
    paper/figures/figure4_governance_scaling.png
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7,
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.2,
    "ytick.major.size":  3.2,
})

# ── Data (Tables 1a & 2 from RESULTS.md) ─────────────────────────────────────
SCALES = [10, 50, 100]   # x-axis for left panel

# Governed metrics across scales
ENTITY_F1  = [0.642, 0.629, 0.612]
TRIPLE_F1  = [0.292, 0.189, 0.133]
TRIPLE_HAL = [0.625, 0.785, 0.796]   # hallucination rate (lower = better)

# Governance-side scorecard at 100 docs
GOV_METRICS = [
    ("Routing exact match",       1.000),
    ("Routing recall",            1.000),
    ("Routing precision",         1.000),
    ("Cross-domain recall",       1.000),
    ("Governance completeness",   1.000),
    ("Audit-trail integrity",     1.000),
    ("Domain coverage",           0.984),
]

# ── Layout ────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.875, 3.0))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.88, wspace=0.48)

# ── Colors ────────────────────────────────────────────────────────────────────
C_ENTITY  = "#2563A8"   # blue — entity F1
C_TRIPLE  = "#DC2626"   # red — triple F1
C_HAL     = "#9CA3AF"   # gray — hallucination rate
C_GOVHIGH = "#16A34A"   # green — governance at ceiling
C_GOVGOOD = "#CA8A04"   # amber — near-ceiling governance

# ═══════════════════════════════════════════════════════
# LEFT PANEL — extraction metrics across scales
# ═══════════════════════════════════════════════════════
X = np.array(SCALES)

ax_left.plot(X, ENTITY_F1, "o-", color=C_ENTITY, lw=1.6, ms=5.5,
             label="Entity F1", zorder=4)
ax_left.plot(X, TRIPLE_F1, "s--", color=C_TRIPLE, lw=1.6, ms=5.0,
             label="Triple F1", zorder=4)

# Secondary axis for hallucination rate
ax_hal = ax_left.twinx()
ax_hal.spines["top"].set_visible(False)
ax_hal.plot(X, TRIPLE_HAL, "^:", color=C_HAL, lw=1.3, ms=4.5,
            label="Triple halluc. rate  (↑ worse)", zorder=3)
ax_hal.set_ylabel("Hallucination rate  (↑ worse)", fontsize=7.5,
                  color=C_HAL, labelpad=4)
ax_hal.tick_params(axis="y", labelsize=7, colors=C_HAL)
ax_hal.set_ylim(0, 1.1)
ax_hal.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax_hal.spines["right"].set_linewidth(0.6)
ax_hal.spines["right"].set_color("#D1D5DB")

# Data point labels
WB = dict(boxstyle="round,pad=0.15", fc="white", ec="none")
for x, ef, tf, hf in zip(X, ENTITY_F1, TRIPLE_F1, TRIPLE_HAL):
    ax_left.text(x, ef + 0.022, f"{ef:.3f}", ha="center", va="bottom",
                 fontsize=6, color=C_ENTITY, bbox=WB)
    ax_left.text(x, tf - 0.030, f"{tf:.3f}", ha="center", va="top",
                 fontsize=6, color=C_TRIPLE, bbox=WB)
    ax_hal.text(x + 2.5, hf, f"{hf*100:.0f}%", ha="left", va="center",
                fontsize=6, color=C_HAL)

# Shaded region showing drift zone on extraction side
ax_left.fill_between(X, TRIPLE_F1, ENTITY_F1, alpha=0.07,
                     color="#2563A8", zorder=1)

# "Extraction drifts" annotation arrow
ax_left.annotate("",
    xy=(100, TRIPLE_F1[2] + 0.015), xytext=(100, TRIPLE_F1[1]),
    arrowprops=dict(arrowstyle="->", color=C_TRIPLE, lw=1.0, mutation_scale=8),
    zorder=5)
ax_left.text(97, (TRIPLE_F1[1] + TRIPLE_F1[2]) / 2 + 0.010,
             "extraction\ndrifts ↓", fontsize=5.5, color=C_TRIPLE,
             ha="right", va="center", style="italic")

ax_left.set_xlim(2, 112)
ax_left.set_ylim(0.0, 0.85)
ax_left.set_xticks(SCALES)
ax_left.set_xticklabels(["10 docs", "50 docs", "100 docs"], fontsize=8)
ax_left.set_ylabel("F1 score", fontsize=8.5, labelpad=4)
ax_left.set_title("Extraction Quality vs. Corpus Scale", fontsize=9, pad=5)
ax_left.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax_left.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_left.set_axisbelow(True)

# Combined legend
h_ent = mpatches.Patch(facecolor=C_ENTITY, edgecolor="none", label="Entity F1")
h_tri = mpatches.Patch(facecolor=C_TRIPLE, edgecolor="none", label="Triple F1")
h_hal = mpatches.Patch(facecolor=C_HAL,   edgecolor="none", label="Triple halluc. rate (right axis)")
ax_left.legend(handles=[h_ent, h_tri, h_hal],
               fontsize=5.8, loc="upper right",
               framealpha=0.92, edgecolor="#D1D5DB",
               borderpad=0.5, labelspacing=0.3, handlelength=0.9)

# ═══════════════════════════════════════════════════════
# RIGHT PANEL — governance scorecard at 100 docs
# ═══════════════════════════════════════════════════════
labels = [m[0] for m in GOV_METRICS]
values = [m[1] for m in GOV_METRICS]
y_pos  = np.arange(len(labels))

colors = [C_GOVHIGH if v >= 0.999 else C_GOVGOOD for v in values]
bars = ax_right.barh(y_pos, values, height=0.55,
                     color=colors, edgecolor="white", linewidth=0.5, zorder=3)

# Value labels at end of bars
for b, v in zip(bars, values):
    ax_right.text(v + 0.002, b.get_y() + b.get_height() / 2,
                  f"{v:.3f}",
                  ha="left", va="center", fontsize=7, fontweight="bold",
                  color="#374151")

# Threshold reference line
ax_right.axvline(0.98, color="#F97316", lw=0.9, ls="--", zorder=2, alpha=0.8)
ax_right.text(0.981, len(labels) - 0.30, "0.98",
              fontsize=5.5, color="#C2410C", ha="left", va="center", style="italic")

ax_right.set_xlim(0.45, 1.08)
ax_right.set_ylim(-0.5, len(labels) - 0.5)
ax_right.set_yticks(y_pos)
ax_right.set_yticklabels(labels, fontsize=7.2)
ax_right.set_xlabel("Score", fontsize=8.5, labelpad=4)
ax_right.set_title("Governance Metrics at 100 docs", fontsize=9, pad=5)
ax_right.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_right.xaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_right.set_axisbelow(True)
ax_right.spines["left"].set_visible(False)

# Legend for right panel
h_ceil = mpatches.Patch(facecolor=C_GOVHIGH, edgecolor="none", label="= 1.000 (exact)")
h_near = mpatches.Patch(facecolor=C_GOVGOOD, edgecolor="none", label="≥ 0.98")
ax_right.legend(handles=[h_ceil, h_near],
                fontsize=6, loc="lower right",
                framealpha=0.92, edgecolor="#D1D5DB",
                borderpad=0.4, labelspacing=0.3, handlelength=0.9)

# ── Save ─────────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "figure4_governance_scaling.pdf"
png_path = OUT_DIR / "figure4_governance_scaling.png"
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
