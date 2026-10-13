"""Regenerate the manuscript data figures (Figs 2-4) from the compact publication summaries.

Run `python scripts/make_publication_summaries.py publication/data/final_v1.parquet` first if the
summaries need to be rebuilt from the released dataset. Fig. 1 (framework diagram) is a TikZ source,
`fig1_framework.tex`, compiled separately with pdflatex.

Colours follow a validated categorical order (blue, orange, aqua); identity is also carried by marker
shape and direct labels, so no series is distinguished by colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "publication" / "results"
CI = pd.read_csv(RES / "final_learned_curve_ci.csv")
SUB = pd.read_csv(RES / "final_context_family_qeubo_t20.csv")

INK, MUTED, GRID = "#2b2b28", "#6b6b65", "#e4e3de"
SERIES = {  # method: (label, colour, marker)
    "audiotwin_qeubo": ("qEUBO", "#2a78d6", "o"),
    "audiotwin_gprandom": ("GP-random", "#eb6834", "^"),
    "audiotwin_bald": ("BALD", "#1baf7a", "s"),
}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- Fig. 2: learning curves with 95% listener-level bootstrap bands -------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.0))
static = CI[CI.method == "best_static"].iloc[0]
ax.axhspan(static.ci_low, static.ci_high, color="#d9d8d2", alpha=0.6, lw=0, zorder=0)
ax.axhline(static.mean_regret, color=INK, ls=(0, (5, 3)), lw=1.4, zorder=1)
ax.text(51.5, static.mean_regret, "Best static\nreference", va="center", ha="left", color=INK, fontsize=8)
label_y = {}
for method, (label, colour, marker) in SERIES.items():
    g = CI[CI.method == method].sort_values("budget")
    ax.fill_between(g.budget, g.ci_low, g.ci_high, color=colour, alpha=0.14, lw=0, zorder=2)
    ax.plot(g.budget, g.mean_regret, color=colour, lw=2, marker=marker, ms=6,
            markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
    label_y[label] = g.mean_regret.iloc[-1]
for label, y in label_y.items():
    dy = {"qEUBO": -0.004, "GP-random": 0.0, "BALD": 0.006}[label]
    ax.text(51.5, y + dy, label, va="center", ha="left", color=INK, fontsize=8)
ax.set_xticks([0, 5, 10, 20, 30, 50])
ax.set_xlim(-1.5, 50.8)
ax.set_xlabel("Pairwise preference queries")
ax.set_ylabel("Mean normalised regret (lower is better)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig2_learning_curves")

# ---- Fig. 3: context and family effects at T=20 (two panels) --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
panels = [
    ("By listening context", SUB[(SUB.family == "ALL") & (SUB.context != "ALL")],
     "context", {"noisy_conversation": "Noisy conversation", "lecture": "Lecture", "music": "Music"}),
    ("By preference family", SUB[(SUB.context == "ALL") & (SUB.family != "ALL")],
     "family", {"intelligibility": "Intelligibility", "comfort": "Comfort", "balanced": "Balanced"}),
]
for ax, (title, d, col, names) in zip(axes, panels):
    d = d.set_index(col).loc[list(names)]
    y = np.arange(len(d))
    v = d.mean_delta.to_numpy()
    ax.axvline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.errorbar(v, y, xerr=np.vstack([v - d.ci_low, d.ci_high - v]), fmt="o", color="#2a78d6",
                ecolor="#2a78d6", elinewidth=1.6, capsize=3, ms=6, markeredgecolor="white", markeredgewidth=1)
    for yi, (vi, hi) in enumerate(zip(v, d.ci_high)):
        ax.text(max(hi, 0.0) + 0.008, yi, f"{vi:+.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, list(names.values()))
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
for ax in axes:
    ax.set_xlabel("qEUBO (T=20) minus best static regret")
axes[0].set_xlim(-0.25, 0.075)
fig.text(0.5, -0.06, "Negative values favour qEUBO. Bars: 95% listener-level bootstrap intervals (N = 128).",
         ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout()
save(fig, "fig3_effects")

# ---- Fig. 4: context x family heatmap (diverging blue-grey-red, symmetric) --------------------
cells = SUB[(SUB.context != "ALL") & (SUB.family != "ALL")]
ctx_order = ["noisy_conversation", "lecture", "music"]
fam_order = ["intelligibility", "comfort", "balanced"]
M = np.array([[cells[(cells.context == c) & (cells.family == f)].mean_delta.iloc[0] for f in fam_order]
              for c in ctx_order])
lim = float(np.ceil(np.abs(M).max() * 20) / 20)
cmap = LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c23634"])
fig, ax = plt.subplots(figsize=(5.6, 3.0))
im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
ax.set_xticks(range(3), ["Intelligibility", "Comfort", "Balanced"])
ax.set_yticks(range(3), ["Noisy conversation", "Lecture", "Music"])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=9,
                color="white" if abs(M[i, j]) > 0.6 * lim else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("qEUBO (T=20) minus best static", fontsize=8)
cb.outline.set_visible(False)
save(fig, "fig4_context_family_heatmap")
print("wrote fig2_learning_curves, fig3_effects, fig4_context_family_heatmap")
