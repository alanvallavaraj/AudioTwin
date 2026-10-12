"""Regenerate the four manuscript figures from frozen compact result summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CURVE = pd.read_csv(ROOT / "publication/results/final_learned_curve_means.csv")
SUB = pd.read_csv(ROOT / "publication/results/final_context_family_qeubo_t20.csv")
STATIC = 0.3212172955694639

labels = {"audiotwin_qeubo": "qEUBO", "audiotwin_bald": "BALD", "audiotwin_gprandom": "GP-random"}
markers = {"audiotwin_qeubo": "o", "audiotwin_bald": "s", "audiotwin_gprandom": "^"}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for method, group in CURVE.groupby("method"):
    group = group.sort_values("budget")
    ax.plot(group.budget, group.mean_regret, marker=markers[method], linewidth=1.8, label=labels[method])
ax.axhline(STATIC, linestyle="--", linewidth=1.5, label="Best static comparator")
ax.set_xlabel("Pairwise preference queries")
ax.set_ylabel("Mean normalised regret (lower is better)")
ax.set_xticks([0, 5, 10, 20, 30, 50])
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(HERE / "fig1_learning_curves.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig1_learning_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)

contexts = SUB[(SUB.family == "ALL") & (SUB.context != "ALL")].copy()
contexts["label"] = contexts.context.map({"noisy_conversation": "Noisy conversation", "lecture": "Lecture", "music": "Music"})
fig, ax = plt.subplots(figsize=(6.8, 3.8))
y = np.arange(len(contexts)); v = contexts.mean_delta.to_numpy()
ax.errorbar(v, y, xerr=np.vstack([v-contexts.ci_low.to_numpy(), contexts.ci_high.to_numpy()-v]), fmt="o", capsize=4, linewidth=1.6)
ax.axvline(0, linewidth=1, linestyle="--")
ax.set_yticks(y, contexts.label); ax.invert_yaxis()
ax.set_xlabel("qEUBO T=20 - best static regret (negative favours qEUBO)")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(HERE / "fig2_context_effects.pdf", bbox_inches="tight"); fig.savefig(HERE / "fig2_context_effects.png", dpi=300, bbox_inches="tight"); plt.close(fig)

families = SUB[(SUB.context == "ALL") & (SUB.family != "ALL")].copy(); families["label"] = families.family.str.capitalize()
fig, ax = plt.subplots(figsize=(6.8, 3.8))
y = np.arange(len(families)); v = families.mean_delta.to_numpy()
ax.errorbar(v, y, xerr=np.vstack([v-families.ci_low.to_numpy(), families.ci_high.to_numpy()-v]), fmt="o", capsize=4, linewidth=1.6)
ax.axvline(0, linewidth=1, linestyle="--")
ax.set_yticks(y, families.label); ax.invert_yaxis()
ax.set_xlabel("qEUBO T=20 - best static regret (negative favours qEUBO)")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(HERE / "fig3_family_effects.pdf", bbox_inches="tight"); fig.savefig(HERE / "fig3_family_effects.png", dpi=300, bbox_inches="tight"); plt.close(fig)

cells = SUB[(SUB.context != "ALL") & (SUB.family != "ALL")]
ctx_order = ["noisy_conversation", "lecture", "music"]
fam_order = ["intelligibility", "comfort", "balanced"]
M = np.array([[cells[(cells.context == c) & (cells.family == f)].mean_delta.iloc[0] for f in fam_order] for c in ctx_order])
lim = max(abs(M.min()), abs(M.max()))
fig, ax = plt.subplots(figsize=(6.8, 3.8))
im = ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
ax.set_xticks(range(3), ["Intelligibility", "Comfort", "Balanced"])
ax.set_yticks(range(3), ["Noisy conversation", "Lecture", "Music"])
for i in range(3):
    for j in range(3): ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax).set_label("qEUBO T=20 - best static regret")
fig.tight_layout(); fig.savefig(HERE / "fig4_context_family_heatmap.pdf", bbox_inches="tight"); fig.savefig(HERE / "fig4_context_family_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)
