"""
Track A PCA: z-score the audited conductance/reversal-potential feature
matrix, run PCA, inspect explained variance + loadings, plot PC1 vs PC2.

Input: track_a_features.csv (produced by track_a_cv_audit.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca(csv_path: str = "track_a_features.csv"):
    df = pd.read_csv(csv_path, index_col="cell_id")
    print(f"Loaded {df.shape[0]} cells x {df.shape[1]} features: {list(df.columns)}\n")

    # --- z-score ---
    scaler = StandardScaler()
    X_z = scaler.fit_transform(df.values)

    # --- PCA ---
    pca = PCA()
    scores = pca.fit_transform(X_z)  # n_cells x n_features

    scores_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(scores.shape[1])],
    )

    # --- explained variance ---
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    print("Explained variance ratio per PC:")
    for i, (v, c) in enumerate(zip(var_ratio, cum_var)):
        print(f"  PC{i+1}: {v:.1%}  (cumulative {c:.1%})")

    # --- loadings: which original features drive each PC ---
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    )
    print("\nLoadings (top contributors to PC1 and PC2):")
    print(loadings[["PC1", "PC2"]].reindex(
        loadings["PC1"].abs().sort_values(ascending=False).index
    ).to_string())

    # --- scree plot + PC1 vs PC2 scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(range(1, len(var_ratio) + 1), var_ratio, color="steelblue")
    axes[0].plot(range(1, len(var_ratio) + 1), cum_var, "o-", color="darkorange", label="cumulative")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Scree plot")
    axes[0].legend()

    axes[1].scatter(scores_df["PC1"], scores_df["PC2"], s=40, edgecolor="black", alpha=0.8)
    for cell_id, row in scores_df.iterrows():
        axes[1].annotate(cell_id, (row["PC1"], row["PC2"]), fontsize=6, alpha=0.6,
                          xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    axes[1].set_title("Cells in PC1-PC2 space")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0, color="gray", lw=0.5)

    plt.tight_layout()
    plt.savefig("track_a_pca_plot.png", dpi=150)
    plt.show()
    print("\nSaved -> track_a_pca_plot.png")

    # --- save scores + loadings for downstream clustering ---
    scores_df.to_csv("track_a_pca_scores.csv")
    loadings.to_csv("track_a_pca_loadings.csv")
    print("Saved -> track_a_pca_scores.csv, track_a_pca_loadings.csv")

    return pca, scores_df, loadings


if __name__ == "__main__":
    run_pca()