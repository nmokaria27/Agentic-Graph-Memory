"""
Figure 5: Governance Operating Modes — Strict vs. Triage
=========================================================
The governance function γ can operate in two modes:
  · audit_only (triage): admits 96% of correct triples, rejects 84% of corrupted
  · strict: admits 56% of correct triples, rejects 100% of corrupted (0% false accepts)

This is a classic precision-recall tradeoff applied to admission control.

Three-element figure:
  Left  — Side-by-side comparison bars: accept recall + reject recall
  Center — False-accept rate bar (lower is better)
  Right — "operating mode" conceptual positioning scatter:
           x = accept recall (good triples in), y = reject recall (bad triples out)
           The ideal corner is (1.0, 1.0).

Data source: paper/RESULTS.md, Table 3.

Run:
    python paper/figures/figure5_strict_vs_triage.py
Outputs:
    paper/figures/figure5_strict_vs_triage.pdf
    paper/figures/figure5_strict_vs_triage.png
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

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
    "legend.fontsize":   7.5,
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.2,
    "ytick.major.size":  3.2,
})

# ── Data (Table 3, RESULTS.md) ────────────────────────────────────────────────
MODES = ["Triage\n(audit-only)", "Strict"]

ACCEPT_RECALL  = [0.96, 0.56]   # fraction of good triples admitted
REJECT_RECALL  = [0.84, 1.00]   # fraction of bad triples rejected
FALSE_ACCEPT   = [0.16, 0.00]   # fraction of bad triples incorrectly admitted

# ── Colors ────────────────────────────────────────────────────────────────────
C_TRIAGE = "#F97316"   # orange — triage default
C_STRICT = "#1D4ED8"   # blue — strict ceiling
C_IDEAL  = "#FBBF24"   # gold star

# ── Layout ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(6.875, 2.8))
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.88, wspace=0.45)

ax_accept, ax_reject, ax_scatter = axes
colors = [C_TRIAGE, C_STRICT]
labels = ["Triage", "Strict"]
X = np.array([0.0, 1.0])
W = 0.40
WB = [pe.withStroke(linewidth=2, foreground="white")]

# ═══════════════════════════════════════════════════════
# LEFT — Accept recall (good triples IN)
# ═══════════════════════════════════════════════════════
bars = ax_accept.bar(X, ACCEPT_RECALL, W, color=colors,
                     edgecolor="white", linewidth=0.5, zorder=3)
for b, v in zip(bars, ACCEPT_RECALL):
    ax_accept.text(b.get_x() + b.get_width() / 2, v / 2,
                   f"{v:.2f}",
                   ha="center", va="center",
                   fontsize=9.5, fontweight="bold", color="white", zorder=4)

# Annotation: triage ≫ strict for good triples
ax_accept.annotate("",
    xy=(X[1], ACCEPT_RECALL[1] + 0.03), xytext=(X[0], ACCEPT_RECALL[0] - 0.01),
    arrowprops=dict(arrowstyle="->", color="#374151", lw=0.9, mutation_scale=8),
    zorder=5)
ax_accept.text(0.5, (ACCEPT_RECALL[0] + ACCEPT_RECALL[1]) / 2 + 0.06,
               "−40 pp\ncost",
               ha="center", va="bottom", fontsize=6, color="#6B7280",
               style="italic")

ax_accept.set_xlim(-0.45, 1.45)
ax_accept.set_ylim(0.0, 1.25)
ax_accept.set_xticks(X)
ax_accept.set_xticklabels(labels, fontsize=9)
ax_accept.set_ylabel("Accept recall\n(correct triples in)", fontsize=7.8, labelpad=3)
ax_accept.set_title("Good-triple recall\n(↑ better)", fontsize=8.5, pad=4)
ax_accept.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax_accept.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_accept.set_axisbelow(True)

# ═══════════════════════════════════════════════════════
# CENTER — Reject recall (bad triples OUT) + false-accept
# ═══════════════════════════════════════════════════════
bars = ax_reject.bar(X, REJECT_RECALL, W, color=colors,
                     edgecolor="white", linewidth=0.5, zorder=3)
for b, v in zip(bars, REJECT_RECALL):
    col = "white" if v > 0.3 else "#374151"
    ax_reject.text(b.get_x() + b.get_width() / 2, v / 2,
                   f"{v:.2f}",
                   ha="center", va="center",
                   fontsize=9.5, fontweight="bold", color=col, zorder=4)

# Overlay: false-accept as red "error" annotation on the triage bar
fa_triage = FALSE_ACCEPT[0]
ax_reject.annotate("",
    xy=(X[0] + W / 2, 1.0 - fa_triage), xytext=(X[0] + W / 2, 1.0),
    arrowprops=dict(arrowstyle="-[,widthB=1.2", color="#DC2626", lw=0.9),
    zorder=5)
ax_reject.text(X[0] + W / 2 + 0.05, 1.0 - fa_triage / 2,
               f"{fa_triage*100:.0f}%\nfail",
               ha="left", va="center", fontsize=6, color="#DC2626",
               style="italic", path_effects=WB)

# Strict: "0% false accept" badge
ax_reject.text(X[1], 1.045, "0 false\naccepts",
               ha="center", va="bottom", fontsize=6.2, color="#16A34A",
               fontweight="bold", path_effects=WB)

ax_reject.set_xlim(-0.45, 1.45)
ax_reject.set_ylim(0.0, 1.25)
ax_reject.set_xticks(X)
ax_reject.set_xticklabels(labels, fontsize=9)
ax_reject.set_ylabel("Reject recall\n(corrupt triples out)", fontsize=7.8, labelpad=3)
ax_reject.set_title("Bad-triple rejection\n(↑ better)", fontsize=8.5, pad=4)
ax_reject.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax_reject.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_reject.set_axisbelow(True)

# ═══════════════════════════════════════════════════════
# RIGHT — Operating-mode scatter (accept vs reject recall)
# ═══════════════════════════════════════════════════════
for ar, rr, c, lb in zip(ACCEPT_RECALL, REJECT_RECALL, colors, labels):
    ax_scatter.scatter(ar, rr, s=110, c=c, edgecolors=c, linewidths=0.8,
                       zorder=5, alpha=0.93)
    offset_x = 0.025 if lb == "Triage" else -0.025
    ha = "left" if lb == "Triage" else "right"
    ax_scatter.text(ar + offset_x, rr, lb,
                    fontsize=7.5, color=c, ha=ha, va="center",
                    fontweight="bold", path_effects=WB, zorder=6)

# Ideal corner star
ax_scatter.plot(1.0, 1.0, "*", color=C_IDEAL, ms=11, zorder=7, mew=0.6, mec="#B45309")
ax_scatter.text(0.996, 0.970, "ideal", fontsize=6, color="#92400E",
                ha="right", va="top", zorder=7)

# Dashed boundary at 0.98 for "governance strong" region
ax_scatter.axvline(0.98, color="#E5E7EB", lw=0.7, ls="--", zorder=1)
ax_scatter.axhline(0.98, color="#E5E7EB", lw=0.7, ls="--", zorder=1)

ax_scatter.set_xlim(0.40, 1.10)
ax_scatter.set_ylim(0.75, 1.15)
ax_scatter.set_xlabel("Accept recall (good triples in)", fontsize=7.8, labelpad=4)
ax_scatter.set_ylabel("Reject recall (bad triples out)", fontsize=7.8, labelpad=4)
ax_scatter.set_title("Operating Mode Space", fontsize=8.5, pad=4)
ax_scatter.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_scatter.set_yticks([0.8, 0.9, 1.0])
ax_scatter.xaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_scatter.yaxis.grid(True, lw=0.4, ls="--", alpha=0.45, zorder=0)
ax_scatter.set_axisbelow(True)

# ── Save ─────────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "figure5_strict_vs_triage.pdf"
png_path = OUT_DIR / "figure5_strict_vs_triage.png"
fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
