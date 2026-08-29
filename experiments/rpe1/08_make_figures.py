"""Generate the three RPE1 manuscript figures from frozen result files.

Expected inputs in results/rpe1:
  replogle_final_exact_integral_grn_v2.npz
  replogle_primary_exact_integral_grn.npz
  replogle_manuscript_validation.csv

No model fitting is performed here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "rpe1"
FIGURES = ROOT / "figures" / "rpe1"
FIGURES.mkdir(parents=True, exist_ok=True)


def figure_identifiability():
    z = np.load(RESULTS / "replogle_final_exact_integral_grn_v2.npz")
    path = pd.DataFrame({
        "k": z["ident_path_k"],
        "rep": z["ident_path_rep"],
        "rank": z["ident_path_rank"],
        "lambda_min": z["ident_path_lambda_min"],
    })
    first = np.asarray(z["ident_first_full_rank"], dtype=float)

    rank_summary = path.groupby("k")["rank"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    lam_summary = path.groupby("k")["lambda_min"].median()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = rank_summary.index.to_numpy()
    axes[0].plot(x, rank_summary["median"].to_numpy())
    axes[0].fill_between(
        x,
        rank_summary["q25"].to_numpy(),
        rank_summary["q75"].to_numpy(),
        alpha=0.2,
    )
    axes[0].axhline(152, linestyle="--")
    axes[0].axvline(np.median(first), linestyle="--")
    axes[0].set_xlabel("Number of perturbation conditions")
    axes[0].set_ylabel("Scaled design rank")
    axes[0].set_title("(a) Empirical rank")

    axes[1].plot(lam_summary.index.to_numpy(), lam_summary.to_numpy())
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of perturbation conditions")
    axes[1].set_ylabel(r"Smallest information eigenvalue $\lambda_{\min}$")
    axes[1].set_title("(b) Conditioning")

    fig.tight_layout()
    fig.savefig(FIGURES / "rpe1_identifiability.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_stability():
    z = np.load(RESULTS / "replogle_primary_exact_integral_grn.npz")
    A = z["A_primary"]
    freq = z["selection_freq_primary"]
    selected_freq = freq[A != 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(selected_freq, bins=np.linspace(0, 1, 21))
    ax.axvline(0.8, linestyle="--")
    n80 = int(np.sum(selected_freq >= 0.8))
    ax.text(
        0.03, 0.95,
        f"{n80:,} selected edges have frequency >= 0.8",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Number of full-data selected edges")
    fig.tight_layout()
    fig.savefig(FIGURES / "rpe1_stability.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_generalization():
    df = pd.read_csv(RESULTS / "replogle_manuscript_validation.csv")
    x = df["r2_phase"].to_numpy()
    y = df["incremental_r2_vs_phase"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=14, alpha=0.7)
    ax.axhline(0.0, linestyle="--")
    rho = df[["r2_phase", "incremental_r2_vs_phase"]].corr(
        method="spearman"
    ).iloc[0, 1]
    win = float(np.mean(y > 0))
    ax.text(
        0.03, 0.95,
        rf"Spearman $\rho$={rho:.3f}" + "\n" + f"GRN better: {100*win:.1f}%",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel(r"Phase-only held-out $R^2$")
    ax.set_ylabel(
        r"$\Delta R^2_{\mathrm{phase}}"
        r"=1-\mathrm{SSE}_{\mathrm{GRN}}/\mathrm{SSE}_{\mathrm{phase}}$"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "rpe1_generalization.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_identifiability()
    figure_stability()
    figure_generalization()
    print(f"Figures written to {FIGURES}")
