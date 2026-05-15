"""
Figure 3: Governance as a Structural Intervention
===================================================
Two panels:

Left panel — Jaccard overlap between governed and flat graphs at 50 docs.
  Two bars: entity Jaccard (0.677) and triple Jaccard (0.180).
  The low triple Jaccard is the key finding: governance is not a thin filter.

Right panel — Where triples come from (50 docs).
  Stacked bar showing:
    · Shared (in both graphs): intersection
    · Governed-only: admitted by governance, never proposed by flat
    · Flat-only: flat admitted, governance rejected
  Overlay: triage board decision distribution (auto-approve vs reviewed).

Data source: paper/RESULTS.md, Tables 1b and 1c.

Run:
    python paper/figures/figure3_structural_impact.py
Outputs:
    paper/figures/figure3_structural_impact.pdf
    paper/figures/figure3_structural_impact.png
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

# ── Data (Table 1b & 1c in RESULTS.md) ───────────────────────────────────────

# Jaccard similarities (50 docs)
ENTITY_JACCARD = 0.677
TRIPLE_JACCARD = 0.180

# Triple counts (50 docs)
# Governed: 274 total, Flat: 407 total
# Shared triples ≈ governed_total * triple_jaccard / (1 + triple_jaccard - triple_jaccard)
# Jaccard = shared / (gov + flat - shared) → shared = J * (gov + flat) / (1 + J)
# Actually from RESULTS.md: unique_to_governed=170, unique_to_flat=303
# shared = governed_total - unique_to_governed = 274 - 170 = 104
# (also: flat_total - unique_to_flat = 407 - 303 = 104 ✓)
SHARED_TRIPLES     = 104
GOVERNED_ONLY      = 170
FLAT_ONLY          = 303

# Triage board activity (50 docs, Table 1c)
AUTO_APPROVED      = 163
REVIEWED           = 130   # escalated to LLM review

# ── Layout ────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.875, 2.9))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.88, wspace=0.40)

# ── Colors ────────────────────────────────────────────────────────────────────
C_ENTITY  = "#93C5FD"   # light blue for entity Jaccard
C_TRIPLE  = "#1D4ED8"   # dark blue for triple Jaccard
C_SHARED  = "#86EFAC"   # green — shared triples
C_GOV     = "#F97316"   # orange — governed-only
C_FLAT    = "#9CA3AF"   # gray — flat-only
C_AUTO    = "#34D399"   # teal — auto-approved
C_REVIEW  = "#FCD34D"   # amber — reviewed

# ═══════════════════════════════════════════════════════
# LEFT PANEL — Jaccard overlap
# ═══════════════════════════════════════════════════════
bar_x = np.array([0.0, 1.0])
bar_h = [ENTITY_JACCARD, TRIPLE_JACCARD]
bar_colors = [C_ENTITY, C_TRIPLE]
bar_labels = ["Entity\noverlap", "Triple\noverlap"]

bars = ax_left.bar(bar_x, bar_h, width=0.45,
                   color=bar_colors, edgecolor="white", linewidth=0.5,
                   zorder=3)

# Value labels inside bars
for b, v in zip(bars, bar_h):
    ax_left.text(b.get_x() + b.get_width() / 2, v / 2,
                 f"{v:.3f}",
                 ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white", zorder=4)

# Annotation emphasising the gap
ax_left.annotate("",
    xy=(1.0, TRIPLE_JACCARD + 0.025), xytext=(0.0, ENTITY_JACCARD - 0.010),
    arrowprops=dict(arrowstyle="-[,widthB=0.6", color="#1D4ED8", lw=0.8),
    zorder=5)
ax_left.text(0.50, 0.50, "≈80% of\ntriples differ",
             ha="center", va="center", fontsize=6.8, color="#1D4ED8",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BFDBFE", lw=0.7),
             zorder=6)

ax_left.set_xlim(-0.45, 1.45)
ax_left.set_ylim(0.0, 1.0)
ax_left.set_xticks(bar_x)
ax_left.set_xticklabels(bar_labels, fontsize=8.5)
ax_left.set_ylabel("Jaccard similarity", fontsize=8.5, labelpad=4)
ax_left.set_title("Set Overlap (50 docs)", fontsize=9, pad=5)
ax_left.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_left.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_left.set_axisbelow(True)

# ═══════════════════════════════════════════════════════
# RIGHT PANEL — Triple origin breakdown + triage overlay
# ═══════════════════════════════════════════════════════

# Left grouped bar: triple distribution (3 categories stacked)
# We show two columns: what's in governed KG vs what's in flat KG
governed_total = SHARED_TRIPLES + GOVERNED_ONLY
flat_total     = SHARED_TRIPLES + FLAT_ONLY

BX = np.array([0.0, 1.0])
W  = 0.38

# Stacked bar for governed KG
ax_right.bar(BX[0], SHARED_TRIPLES, W,
             color=C_SHARED, edgecolor="white", linewidth=0.5, zorder=3,
             label=f"Shared  ({SHARED_TRIPLES})")
ax_right.bar(BX[0], GOVERNED_ONLY, W, bottom=SHARED_TRIPLES,
             color=C_GOV, edgecolor="white", linewidth=0.5, zorder=3,
             label=f"Governed-only  ({GOVERNED_ONLY})")

# Stacked bar for flat KG
ax_right.bar(BX[1], SHARED_TRIPLES, W,
             color=C_SHARED, edgecolor="white", linewidth=0.5, zorder=3)
ax_right.bar(BX[1], FLAT_ONLY, W, bottom=SHARED_TRIPLES,
             color=C_FLAT, edgecolor="white", linewidth=0.5, zorder=3,
             label=f"Flat-only  ({FLAT_ONLY})")

# Value labels: total
for bx, total in [(BX[0], governed_total), (BX[1], flat_total)]:
    ax_right.text(bx, total + 8, str(total),
                  ha="center", va="bottom", fontsize=7.5,
                  fontweight="bold", color="#374151")

# Triage bar on a secondary axis (inset)
ax2 = ax_right.twinx()
ax2.spines["top"].set_visible(False)
TRIAGE_X = 1.55
TOTAL_TRIAGE = AUTO_APPROVED + REVIEWED
ax2.bar(TRIAGE_X, AUTO_APPROVED / TOTAL_TRIAGE, 0.30,
        color=C_AUTO, edgecolor="white", linewidth=0.5, zorder=3,
        label=f"Auto-approved  ({AUTO_APPROVED})")
ax2.bar(TRIAGE_X, REVIEWED / TOTAL_TRIAGE, 0.30,
        bottom=AUTO_APPROVED / TOTAL_TRIAGE,
        color=C_REVIEW, edgecolor="white", linewidth=0.5, zorder=3,
        label=f"LLM-reviewed  ({REVIEWED})")
ax2.set_ylabel("Triage fraction", fontsize=7.5, labelpad=4, color="#6B7280")
ax2.tick_params(axis="y", labelsize=7, colors="#6B7280")
ax2.set_ylim(0, 1.45)
ax2.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
ax2.yaxis.set_tick_params(width=0.6)
ax2.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=6.5, color="#6B7280")
ax2.text(TRIAGE_X, 1.04, "Triage\nboard",
         ha="center", va="bottom", fontsize=6.2, color="#6B7280", style="italic")
ax2.spines["right"].set_linewidth(0.6)
ax2.spines["right"].set_color("#D1D5DB")

ax_right.set_xlim(-0.40, 1.80)
ax_right.set_ylim(0, 560)
ax_right.set_xticks([BX[0], BX[1]])
ax_right.set_xticklabels(["Governed\nKG", "Flat\nKG"], fontsize=8.5)
ax_right.set_ylabel("Triple count", fontsize=8.5, labelpad=4)
ax_right.set_title("Triple Origin (50 docs)", fontsize=9, pad=5)
ax_right.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_right.set_axisbelow(True)

# Combined legend (right panel)
h1 = mpatches.Patch(facecolor=C_SHARED, edgecolor="none", label=f"Shared  ({SHARED_TRIPLES})")
h2 = mpatches.Patch(facecolor=C_GOV,    edgecolor="none", label=f"Governed-only  ({GOVERNED_ONLY})")
h3 = mpatches.Patch(facecolor=C_FLAT,   edgecolor="none", label=f"Flat-only  ({FLAT_ONLY})")
h4 = mpatches.Patch(facecolor=C_AUTO,   edgecolor="none", label=f"Auto-approved  ({AUTO_APPROVED})")
h5 = mpatches.Patch(facecolor=C_REVIEW, edgecolor="none", label=f"LLM-reviewed  ({REVIEWED})")
ax_right.legend(handles=[h1, h2, h3, h4, h5],
                fontsize=5.8, loc="upper center",
                bbox_to_anchor=(0.42, -0.14),
                ncol=3, framealpha=0.92, edgecolor="#D1D5DB",
                borderpad=0.5, columnspacing=0.6, handlelength=0.9)

# ── Save ─────────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "figure3_structural_impact.pdf"
png_path = OUT_DIR / "figure3_structural_impact.png"
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
