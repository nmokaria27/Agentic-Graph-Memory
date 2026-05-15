"""
Figure 1: System Architecture — Paper Style
=============================================
Redesigned to match typical NLP/ML paper figures:
  - Light fills + dark text (readable at print scale)
  - Horizontal section headers (not rotated)
  - Minimal, precise arrows
  - Consistent box sizing within rows
  - Adequate whitespace

Run:
    python paper/figures/figure1_architecture.py
Outputs:
    paper/figures/figure1_architecture.pdf
    paper/figures/figure1_architecture.png
"""

import pathlib
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":   8,
    "text.usetex": False,
    "figure.dpi":  300,
})

fig, ax = plt.subplots(figsize=(6.875, 5.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Palette: light fill / dark border / dark text ────────────────────────
B_F, B_E, B_T = "#DBEAFE", "#2563A8", "#1E3A8A"   # blue  — workers
P_F, P_E, P_T = "#EDE9FE", "#6D28D9", "#4C1D95"   # purple — deliberation
O_F, O_E, O_T = "#FFEDD5", "#C2410C", "#7C2D12"   # orange — coordinators
G_F, G_E, G_T = "#F3F4F6", "#6B7280", "#374151"   # gray  — shared memory
A1F, A1E      = "#FEF9C3", "#CA8A04"               # amber1 — propose
A2F, A2E      = "#FDE68A", "#B45309"               # amber2 — route
RF,  RE, RT   = "#FEE2E2", "#DC2626", "#7F1D1D"   # red   — triage board
CF,  CE, CT   = "#FCE7F3", "#BE185D", "#831843"   # crimson — commit
GRF, GRE, GRT = "#DCFCE7", "#16A34A", "#14532D"   # green — governed KG
TF,  TE, TT   = "#CCFBF1", "#0D9488", "#134E4A"   # teal  — application
ARR            = "#374151"                          # arrow color


# ── Primitives ───────────────────────────────────────────────────────────

def rect(cx, cy, w, h, fc, ec, lw=1.2, ls="-", zorder=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=zorder,
    ))


def txt(x, y, s, fs=7, color="#111827", bold=False,
        ha="center", va="center", italic=False, zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal",
            zorder=zorder, multialignment="center")


def box(cx, cy, w, h, fc, ec, label, tc, fs=6.5, lw=1.2):
    rect(cx, cy, w, h, fc, ec, lw=lw)
    txt(cx, cy, label, fs=fs, color=tc, bold=True)


def arr(x0, y0, x1, y1, c=ARR, lw=1.0, style="->", rad=0.0, ms=9):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=ms),
                zorder=4)


def section_band(y0, h, fc, ec, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (0.015, y0), 0.970, h,
        boxstyle="round,pad=0.006",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1,
    ))


# ══════════════════════════════════════════════════════════════════════════
# CREATION LAYER   y: 0.525 – 0.975
# ══════════════════════════════════════════════════════════════════════════
section_band(0.525, 0.450, "#F8FAFF", B_E, lw=1.2)
txt(0.038, 0.960, "Creation Layer", fs=7.5, color=B_T, bold=True,
    ha="left", va="top")

# ── Input ──
txt(0.50, 0.990, "Document Corpus", fs=9.5, color="#111827", bold=True)
arr(0.50, 0.983, 0.50, 0.970)

# ── Worker agents (stages 1–5)  y = 0.895 ──
WY, WW, WH = 0.893, 0.102, 0.055
WXS = [0.130, 0.256, 0.382, 0.508, 0.634]
WLBS = ["Doc\nProcessor", "Domain\nClassifier",
        "Entity\nExtractor", "Relation\nExtractor", "Evidence\nLinker"]

for i, (x, lb) in enumerate(zip(WXS, WLBS)):
    box(x, WY, WW, WH, B_F, B_E, lb, B_T, fs=6.5)
    txt(x, WY + WH/2 + 0.015, str(i + 1), fs=5.5, color=B_E, bold=True)

for i in range(4):
    arr(WXS[i] + WW/2, WY, WXS[i+1] - WW/2, WY, lw=0.9)

# ── Shared Memory sidebar  (dashed box) ──
SM_CX, SM_CY = 0.857, 0.759
SM_W,  SM_H  = 0.122, 0.330

rect(SM_CX, SM_CY, SM_W, SM_H, G_F, G_E, lw=0.9, ls="--")
txt(SM_CX, SM_CY + 0.128, "Shared Memory", fs=7, color=G_T, bold=True)
for i, mt in enumerate(["Episodic", "Semantic", "Working", "Blackboard"]):
    txt(SM_CX, SM_CY + 0.060 - i * 0.056, f"· {mt}", fs=6.2, color="#6B7280")

# Two representative dashed links to shared memory
for x in [WXS[1], WXS[3]]:
    ax.annotate("", xy=(SM_CX - SM_W/2, WY), xytext=(x + WW/2, WY),
                arrowprops=dict(arrowstyle="-", color=G_E, lw=0.6,
                                linestyle="dashed"), zorder=2)

# ── Deliberation (stage 6)  y = 0.790 ──
DX, DY = 0.405, 0.790
DW, DH = 0.248, 0.066

box(DX, DY, DW, DH, P_F, P_E,
    "Deliberation  (Stage 6)\nMulti-Agent Vote & Debate",
    P_T, fs=6.5, lw=1.2)

# Arrow: EvidenceLinker → Deliberation (uncertain branch)
arr(WXS[4], WY - WH/2,
    DX + DW/2 - 0.018, DY + DH/2,
    c=P_E, lw=0.9)
txt(WXS[4] + 0.015, (WY - WH/2 + DY + DH/2)/2 + 0.008,
    "conf 0.35–0.65", fs=5.2, color=P_T, italic=True)

# Dashed: deliberation ↔ shared memory
ax.annotate("", xy=(SM_CX - SM_W/2, DY), xytext=(DX + DW/2, DY),
            arrowprops=dict(arrowstyle="-", color=G_E, lw=0.6,
                            linestyle="dashed"), zorder=2)

# ── Coordinator agents (stages 7–9)  y = 0.663 ──
CY, CW, CH = 0.663, 0.170, 0.058
CXS = [0.193, 0.400, 0.607]
CLBS = ["Extraction\nValidator", "Extraction\nVerification", "Knowledge\nOrganizer"]

for i, (x, lb) in enumerate(zip(CXS, CLBS)):
    box(x, CY, CW, CH, O_F, O_E, lb, O_T, fs=6.5)
    txt(x, CY + CH/2 + 0.015, str(i + 7), fs=5.5, color=O_E, bold=True)

for i in range(2):
    arr(CXS[i] + CW/2, CY, CXS[i+1] - CW/2, CY, lw=0.9)

# Dashed: coordinator → shared memory
ax.annotate("", xy=(SM_CX - SM_W/2, CY), xytext=(CXS[2] + CW/2, CY),
            arrowprops=dict(arrowstyle="-", color=G_E, lw=0.6,
                            linestyle="dashed"), zorder=2)

# Arrow: Deliberation → ExtractionValidator
arr(DX - DW/2 + 0.022, DY - DH/2,
    CXS[0] - 0.008, CY + CH/2,
    c=P_E, lw=0.9)

# High-confidence bypass (thin, gray, curved — from Evidence Linker to ExtrValidator)
arr(WXS[4], WY - WH/2, CXS[0] - CW/2, CY + CH/2,
    c="#9CA3AF", lw=0.75, rad=-0.30)
txt(0.095, 0.775, "conf ≥ 0.65\n(bypass)", fs=5.0, color="#9CA3AF", italic=True)

# Arrow: KnowledgeOrganizer → Governance
arr(CXS[2], CY - CH/2, CXS[2], 0.520)
txt(CXS[2] + 0.040, 0.542, "verified triples", fs=5.2,
    color=ARR, italic=True, ha="left")


# ══════════════════════════════════════════════════════════════════════════
# GOVERNANCE LAYER   y: 0.315 – 0.515
# ══════════════════════════════════════════════════════════════════════════
section_band(0.315, 0.200, "#FFFEF5", A1E, lw=1.1)
txt(0.038, 0.502, "Governance Layer", fs=7.5, color="#78350F", bold=True,
    ha="left", va="top")

GY, GW, GH = 0.415, 0.168, 0.068

gov = [
    (0.185, "Propose\nTriple",        A1F, A1E, "#78350F"),
    (0.365, "Route\n(φ: ownership)", A2F, A2E, "#92400E"),
    (0.555, "Triage\nBoard",          RF,  RE,  RT),
    (0.745, "Commit\n+ Audit Log",    CF,  CE,  CT),
]
for x, lb, fc, ec, tc in gov:
    box(x, GY, GW, GH, fc, ec, lb, tc, fs=6.5)

for i in range(3):
    arr(gov[i][0] + GW/2, GY, gov[i+1][0] - GW/2, GY,
        c="#92400E", lw=0.9)

# Triage outcome labels — stacked inside the governance band
ty = GY - GH/2 - 0.010
for j, (lbl, col) in enumerate([
    ("auto-approve",        "#166534"),
    ("cross-domain review", "#1D4ED8"),
    ("revise / reject",     "#991B1B"),
]):
    txt(gov[2][0], ty - j * 0.024, lbl,
        fs=5.2, color=col, italic=True)

# Arrow: Commit → KG  (lands at top of KG band)
arr(gov[3][0], GY - GH/2, gov[3][0], 0.308)
txt(gov[3][0] - 0.070, 0.330, "admitted\ntriples",
    fs=5.2, color=ARR, italic=True)


# ══════════════════════════════════════════════════════════════════════════
# GOVERNED KG   y: 0.208 – 0.308
# ══════════════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch(
    (0.015, 0.208), 0.970, 0.098,
    boxstyle="round,pad=0.008",
    facecolor=GRF, edgecolor=GRE, linewidth=2.0, zorder=2,
))
txt(0.50, 0.268, "Governed Knowledge Graph",
    fs=11.5, color=GRT, bold=True)
txt(0.50, 0.233,
    "Entities  ·  Triples  ·  Domain Org Chart  ·  Audit Log",
    fs=7, color="#166534")

# Arrow: KG → Application
arr(0.358, 0.208, 0.358, 0.192)


# ══════════════════════════════════════════════════════════════════════════
# APPLICATION LAYER   y: 0.025 – 0.202
# ══════════════════════════════════════════════════════════════════════════
section_band(0.025, 0.178, "#F0FDFA", TE, lw=1.1)
txt(0.038, 0.190, "Application Layer", fs=7.5, color=TT, bold=True,
    ha="left", va="top")

AY, AW, AH = 0.105, 0.170, 0.062
app = [
    (0.170, "User\nQuery"),
    (0.358, "QA\nOrchestrator"),
    (0.575, "Domain Expert\nRouting"),
    (0.790, "Synthesized\nAnswer"),
]
for x, lb in app:
    box(x, AY, AW, AH, TF, TE, lb, TT, fs=6.5)

for i in range(3):
    arr(app[i][0] + AW/2, AY, app[i+1][0] - AW/2, AY, c=TE, lw=0.9)

# Arrow: KG → QA Orchestrator
arr(0.358, 0.208, 0.358, AY + AH/2)

# Double-headed: Domain Expert Routing ↔ KG
arr(app[2][0] + 0.012, AY + AH/2, 0.638, 0.208,
    c=GRE, lw=0.9, style="<->")
txt(0.668, 0.177, "subgraph retrieval", fs=5.2, color=GRT, italic=True,
    ha="left")


# ══════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════
legend = [
    (B_F,  B_E,  "Worker agents (1–5)"),
    (P_F,  P_E,  "Deliberation (Stage 6)"),
    (O_F,  O_E,  "Coordinator agents (7–9)"),
    (G_F,  G_E,  "Shared Memory (dashed)"),
]
for i, (fc, ec, lb) in enumerate(legend):
    lx = 0.055 + i * 0.233
    ax.add_patch(FancyBboxPatch(
        (lx, 0.003), 0.013, 0.011,
        boxstyle="round,pad=0.002",
        facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=5,
    ))
    txt(lx + 0.017, 0.0085, lb, fs=5.8, color="#374151",
        ha="left", va="center", bold=False)


# ── Save ─────────────────────────────────────────────────────────────────
fig.savefig(OUT_DIR / "figure1_architecture.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT_DIR / "figure1_architecture.png", bbox_inches="tight", dpi=300)
print(f"Saved:\n  {OUT_DIR / 'figure1_architecture.pdf'}"
      f"\n  {OUT_DIR / 'figure1_architecture.png'}")
plt.show()
