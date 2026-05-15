"""
Figure 2: Governed vs. Flat KG Creation Quality
================================================
Three panels: Entity F1 | Triple F1 | Triple Hallucination Rate
At two scales: 10 docs and 50 docs.

Data source: evaluation/results/paper_table_consolidated.json
All values confirmed against paper/RESULTS.md.

Run:
    python paper/figures/figure2_governed_vs_flat.py
Outputs:
    paper/figures/figure2_governed_vs_flat.pdf
    paper/figures/figure2_governed_vs_flat.png
"""

import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── NeurIPS-style rcParams ───────────────────────────────────────────────────
# NeurIPS body text is 10pt; axis labels slightly smaller.
# Set usetex=True if a LaTeX installation is available for exact CM font match.
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9.5,
    "axes.labelsize":     8,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    8.5,
    "figure.dpi":         300,
    "text.usetex":        False,   # flip to True if LaTeX is on PATH
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
})

# ── Data ─────────────────────────────────────────────────────────────────────
# From evaluation/results/paper_table_consolidated.json + paper/RESULTS.md.
# Indices: [10 docs, 50 docs]

SCALES = ["10 docs", "50 docs"]

METRICS = [
    {
        "title":       "Entity F1",
        "ylabel":      "F1",
        "lower_better": False,
        "flat":        [0.481, 0.633],
        "governed":    [0.642, 0.629],
        "ylim":        (0.0, 0.88),
        "yticks":      [0.0, 0.2, 0.4, 0.6, 0.8],
    },
    {
        "title":       "Triple F1",
        "ylabel":      "F1",
        "lower_better": False,
        "flat":        [0.079, 0.153],
        "governed":    [0.292, 0.189],
        "ylim":        (0.0, 0.48),
        "yticks":      [0.0, 0.1, 0.2, 0.3, 0.4],
    },
    {
        "title":       "Triple Hallucination Rate",
        "ylabel":      "Rate  (↓ better)",
        "lower_better": True,
        "flat":        [0.872, 0.858],
        "governed":    [0.625, 0.785],
        "ylim":        (0.0, 1.12),
        "yticks":      [0.0, 0.25, 0.50, 0.75, 1.00],
    },
]

# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_FLAT     = "#8eb8d4"   # muted steel-blue
COLOR_GOVERNED = "#e07a38"   # warm amber-orange
COLOR_IMPROVE  = "#2e7d32"   # dark green  — governance is better
COLOR_NEUTRAL  = "#666666"   # gray        — essentially no change

# ── Layout ───────────────────────────────────────────────────────────────────
# NeurIPS two-column text width ≈ 6.875 in; height chosen for compactness.
fig, axes = plt.subplots(1, 3, figsize=(6.875, 2.55))
fig.subplots_adjust(wspace=0.40, left=0.07, right=0.97, bottom=0.22, top=0.87)

# Bar geometry
X       = np.array([0.0, 1.0])   # group centres for [10 docs, 50 docs]
W       = 0.30                    # individual bar width
GAP     = 0.04                    # gap between the two bars in a pair
X_FLAT  = X - W / 2 - GAP / 2
X_GOV   = X + W / 2 + GAP / 2


def _delta_label(flat_val: float, gov_val: float, lower_better: bool) -> tuple[str, str]:
    """Return (label_string, color) for the delta annotation."""
    if lower_better:
        delta_pp = (flat_val - gov_val) * 100          # positive = improvement
        if abs(delta_pp) < 0.5:
            return "≈0 pp", COLOR_NEUTRAL
        sign = "−" if delta_pp > 0 else "+"            # − means rate dropped (good)
        return f"{sign}{abs(delta_pp):.1f} pp", COLOR_IMPROVE if delta_pp > 0 else "#c62828"
    else:
        if flat_val == 0:
            return "—", COLOR_NEUTRAL
        rel_pct = (gov_val - flat_val) / flat_val * 100
        if abs(rel_pct) < 1.0:
            return "≈0%", COLOR_NEUTRAL
        sign = "+" if rel_pct > 0 else ""
        return f"{sign}{rel_pct:.0f}% rel", COLOR_IMPROVE if rel_pct > 0 else "#c62828"


for ax, m in zip(axes, METRICS):
    fv = np.array(m["flat"])
    gv = np.array(m["governed"])

    # Bars
    ax.bar(X_FLAT, fv, W, color=COLOR_FLAT,     edgecolor="white",
           linewidth=0.4, zorder=3, label="Flat")
    ax.bar(X_GOV,  gv, W, color=COLOR_GOVERNED, edgecolor="white",
           linewidth=0.4, zorder=3, label="Governed (triage)")

    # Delta annotations centered above each bar pair
    for i in range(len(SCALES)):
        x_ctr  = (X_FLAT[i] + X_GOV[i]) / 2
        y_top  = max(fv[i], gv[i])
        label, color = _delta_label(fv[i], gv[i], m["lower_better"])

        ax.text(
            x_ctr, y_top + (m["ylim"][1] - m["ylim"][0]) * 0.030,
            label,
            ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=color,
        )

    # Value labels inside each bar (only if bar is tall enough)
    for bars_x, vals in [(X_FLAT, fv), (X_GOV, gv)]:
        for bx, v in zip(bars_x, vals):
            if v > m["ylim"][1] * 0.10:
                ax.text(
                    bx, v / 2, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold",
                )

    # Axis formatting
    ax.set_xlim(-0.58, 1.58)
    ax.set_ylim(*m["ylim"])
    ax.set_xticks(X)
    ax.set_xticklabels(SCALES, fontsize=8.5)
    ax.set_ylabel(m["ylabel"], fontsize=8, labelpad=3)
    ax.set_title(m["title"], fontsize=9.5, pad=5)
    ax.set_yticks(m["yticks"])

    if m["lower_better"]:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # Subtle horizontal gridlines
    ax.yaxis.grid(True, linewidth=0.45, linestyle="--", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

# ── Shared legend ─────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=COLOR_FLAT,     edgecolor="none", label="Flat (unguided)"),
    mpatches.Patch(facecolor=COLOR_GOVERNED, edgecolor="none", label="Governed (triage)"),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False,
    fontsize=8.5,
    handlelength=1.2,
    handleheight=0.9,
    columnspacing=1.2,
)

# ── Save ──────────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "figure2_governed_vs_flat.pdf"
png_path = OUT_DIR / "figure2_governed_vs_flat.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
