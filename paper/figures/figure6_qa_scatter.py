"""
Figure 6: QA System Comparison — Faithfulness × Coverage Scatter
=================================================================
Each point is one QA system evaluated on the 50-document governed KG
(n = 14 questions, Table 6 in the paper).

Axes:
  X — Coverage    (fraction of gold-answer content addressed)
  Y — Faithfulness (fraction of atomic claims entailed by the KG)
Point size ∝ Corrected QA score (the composite metric).

Two stories in the figure:
  1. "Graph structure → faithfulness ceiling":
       adding any KG lifts faithfulness from 0.833 (Doc RAG) to 1.000.
  2. "Governance routing → coverage frontier":
       within the faithfulness-ceiling band, governed domain routing
       pushes coverage from 0.864 (Flat KG) to 0.964 (Domain advanced).

Data source: paper/RESULTS.md, Table 6 (paper main body).

Run:
    python paper/figures/figure6_qa_scatter.py
Outputs:
    paper/figures/figure6_qa_scatter.pdf
    paper/figures/figure6_qa_scatter.png
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── NeurIPS rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8.5,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   6.5,
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.2,
    "ytick.major.size":  3.2,
})

# ── Data (Table 6, paper main body) ──────────────────────────────────────
SYSTEMS = [
    dict(label="GraphRAG\n(official)",   family="independent",
         cov=0.357, faith=0.357, corrected_qa=0.238, note="†"),
    dict(label="Doc RAG",                family="baseline",
         cov=0.714, faith=0.833, corrected_qa=0.492, note=""),
    dict(label="GraphRAG\n(community)",  family="graph",
         cov=0.793, faith=1.000, corrected_qa=0.669, note=""),
    dict(label="Flat KG",               family="graph",
         cov=0.864, faith=1.000, corrected_qa=0.680, note=""),
    dict(label="Domain\n(basic)",        family="governed",
         cov=0.950, faith=1.000, corrected_qa=0.693, note=""),
    dict(label="Domain\n(advanced)",     family="governed",
         cov=0.964, faith=1.000, corrected_qa=0.684, note=""),
]

# ── Visual encoding ───────────────────────────────────────────────────────
STYLE = {
    "independent": ("#DC2626", "D", "#7F1D1D"),
    "baseline":    ("#6B7280", "s", "#374151"),
    "graph":       ("#2563A8", "o", "#1E3A8A"),
    "governed":    ("#C05621", "^", "#7C2D12"),
}
SIZE_SCALE = 380   # small enough that the ceiling cluster doesn't occlude labels

# ── Figure: wider than square to give horizontal room ────────────────────
fig, ax = plt.subplots(figsize=(4.6, 3.4))
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.88)

# Y range extended above 1.0 to hold ceiling labels
ax.set_xlim(0.22, 1.07)
ax.set_ylim(0.26, 1.17)

# ── Grid and axes ─────────────────────────────────────────────────────────
ax.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax.xaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("Coverage", fontsize=8.5, labelpad=4)
ax.set_ylabel("Faithfulness", fontsize=8.5, labelpad=4)
ax.set_title("QA Systems: Faithfulness × Coverage", fontsize=9, pad=5)
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# ── Scatter points ────────────────────────────────────────────────────────
for s in SYSTEMS:
    color, marker, edge = STYLE[s["family"]]
    ax.scatter(s["cov"], s["faith"],
               s=s["corrected_qa"] * SIZE_SCALE,
               c=color, marker=marker,
               edgecolors=edge, linewidths=0.8,
               zorder=5, alpha=0.93)

# ── Faithfulness ceiling line ─────────────────────────────────────────────
ax.axhline(1.0, color="#9CA3AF", lw=0.8, ls="--", zorder=1, alpha=0.75)
ax.text(0.24, 1.005, "Faithfulness ceiling",
        fontsize=5.8, color="#6B7280", va="bottom", ha="left",
        style="italic", zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# ── Labels: isolated points (GraphRAG official, Doc RAG) ──────────────────
WB = [pe.withStroke(linewidth=2, foreground="white")]

ax.text(0.357, 0.320, "GraphRAG\n(official)†",
        fontsize=6.2, color="#DC2626", ha="center", va="bottom",
        multialignment="center", zorder=6, path_effects=WB)

ax.text(0.730, 0.810, "Doc RAG",
        fontsize=6.2, color="#6B7280", ha="left", va="top",
        zorder=6, path_effects=WB)

# ── Labels: faithfulness ceiling cluster — placed ABOVE the ceiling line ──
# Alternating heights (1.04 / 1.07) so adjacent labels don't overlap.
# Thin tick lines connect each label to its data point.
CEIL_CFG = [
    # idx  label_x  label_y   ha
    (2,    0.793,   1.040,   "center"),   # GraphRAG community
    (3,    0.864,   1.070,   "center"),   # Flat KG
    (4,    0.940,   1.040,   "center"),   # Domain basic  (shift left of point)
    (5,    1.000,   1.070,   "center"),   # Domain advanced (shift right)
]
for idx, lx, ly, ha in CEIL_CFG:
    s = SYSTEMS[idx]
    color, _, _ = STYLE[s["family"]]
    # dotted tick from data point up to label bottom
    ax.plot([s["cov"], lx], [1.002, ly - 0.010],
            color=color, lw=0.6, ls=":", zorder=4, alpha=0.65)
    ax.text(lx, ly, s["label"] + s["note"],
            fontsize=6.2, color=color, ha=ha, va="bottom",
            multialignment="center", zorder=6, path_effects=WB)

# ── Annotation 1: graph structure → faithfulness ceiling ──────────────────
ax.annotate("",
    xy=(0.780, 0.997), xytext=(0.720, 0.852),
    arrowprops=dict(arrowstyle="->", color="#2563A8", lw=1.0,
                    mutation_scale=8, connectionstyle="arc3,rad=0.25"),
    zorder=6)
ax.text(0.668, 0.924,
        "Graph →\nfaithfulness ↑",
        fontsize=5.5, color="#1D4ED8", ha="right", va="center",
        style="italic", zorder=7,
        path_effects=WB)

# ── Annotation 2: governance routing → coverage frontier ──────────────────
# Arrow along the ceiling (above the dashed line, below the lowest ceiling label)
ax.annotate("",
    xy=(0.936, 1.017), xytext=(0.870, 1.017),
    arrowprops=dict(arrowstyle="->", color="#C05621", lw=1.2,
                    mutation_scale=9),
    zorder=6)
ax.text(0.903, 1.024,
        "Governance → coverage ↑",
        fontsize=5.5, color="#9A3412", ha="center", va="bottom",
        style="italic", zorder=7,
        path_effects=WB)

# ── Ideal corner ──────────────────────────────────────────────────────────
ax.plot(1.0, 1.0, "*", color="#FBBF24", ms=9, zorder=7, mew=0.5, mec="#B45309")
ax.text(0.998, 0.977, "ideal", fontsize=5.5, color="#92400E",
        ha="right", va="top", zorder=7)

# ── Corrected QA size legend (lower-center, clear of data) ───────────────
size_handles = [
    plt.scatter([], [], s=qa * SIZE_SCALE,
                c="#9CA3AF", marker="o",
                edgecolors="#6B7280", linewidths=0.6, alpha=0.85, label=f"{qa:.2f}")
    for qa in [0.40, 0.55, 0.693]
]
size_leg = ax.legend(
    handles=size_handles,
    title="Corrected\nQA score",
    title_fontsize=5.8, fontsize=6,
    loc="lower center", bbox_to_anchor=(0.47, 0.01),
    framealpha=0.92, edgecolor="#D1D5DB",
    borderpad=0.5, labelspacing=0.3, handletextpad=0.4,
    ncol=3, columnspacing=0.6,
)

# ── System family legend (upper left) ─────────────────────────────────────
family_handles = [
    mpatches.Patch(facecolor="#DC2626", edgecolor="#7F1D1D",
                   label="GraphRAG (independent)†"),
    mpatches.Patch(facecolor="#6B7280", edgecolor="#374151",
                   label="Document RAG"),
    mpatches.Patch(facecolor="#2563A8", edgecolor="#1E3A8A",
                   label="Graph-based (no governance)"),
    mpatches.Patch(facecolor="#C05621", edgecolor="#7C2D12",
                   label="Governed (domain routing)"),
]
ax.legend(handles=family_handles, fontsize=5.8, loc="upper left",
          framealpha=0.92, edgecolor="#D1D5DB",
          borderpad=0.5, labelspacing=0.28,
          handlelength=0.9, handleheight=0.8)
ax.add_artist(size_leg)

# ── Footnote ─────────────────────────────────────────────────────────────
fig.text(0.12, 0.003,
         "† Uses an independent graph index, not the governed KG.",
         fontsize=5.2, color="#6B7280", va="bottom", ha="left")

# ── Save ──────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "figure6_qa_scatter.pdf"
png_path = OUT_DIR / "figure6_qa_scatter.png"
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
