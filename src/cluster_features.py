"""Step 7+: exploratory unsupervised clustering over master_features.csv,
to see whether the 69 cells fall into natural groups that might correspond
to pyloric network roles (PD/AB, LP, PY, ...). Hypothesis-generating only --
not validated against any ground-truth cell identity yet (see the
checklist's still-open "network-embedding validation" step), and 69 cells
is a small N for clustering, so treat whatever comes out accordingly.

Two independent views are clustered and cross-checked against each other,
not just one:
  - "raw": standardized behavioral features directly (Euclidean distance
    in the full feature space).
  - "pca": scores on a PCA of that same standardized matrix, which is
    naturally robust to collinear features since PCA orthogonalizes them.
If the two views agree (checked via adjusted Rand index), that's
reassuring; if they diverge, it's worth understanding why rather than
trusting either alone.

Conductances (gNa, gCaT, ...) are excluded from the default "behavioral"
feature set -- they're the generative genotype, not the phenotype a
downstream network would actually see, which is what a pyloric-role
classification should plausibly track. Run with --feature-set conductance
or --feature-set all for a comparison pass.

No feature is dropped for having low total variance: low variance across
all cells doesn't mean a feature is irrelevant for clustering -- it could
be exactly what separates a small, real minority cluster from everything
else, which a variance-ranked cut would erase before ever looking for it.
Three things ARE dropped: features with essentially zero variance
(uninformative, and would blow up under z-scoring); features below
--min-feature-coverage (too many missing cells to trust, same convention
as consolidate_features.py's PCA -- see select_complete_case_matrix
there); and, as of the full 69-cell run, one column from each highly-
correlated pair (see prune_correlated_features). That last one used to be
flag-only, but confirmed directly this was letting redundant features
silently multiply-count the same underlying axis in the raw Euclidean
distance (baseline_freq_hz/peak_test_freq_hz/max_rebound_spike_count, all
mutually r=0.94-0.99, were really one "how excitable is this cell"
dimension counted three times) -- almost certainly why unconstrained
silhouette-based k selection kept climbing to the search boundary (k=7 of
8) instead of settling on anything, despite never exceeding ~0.34 (weak
structure by any conventional read). --k now defaults to 3, matching the
known AB/PD, LP, PY count, rather than trusting silhouette to pick an
unconstrained k blind -- pass --k explicitly to override for exploration.
"""

import pickle
import sys
from pathlib import Path
import argparse
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from consolidate_features import select_complete_case_matrix, DEFAULT_OUTPUT_CSV_PATH as DEFAULT_MASTER_CSV_PATH

DEFAULT_OUTPUT_CSV_PATH = ROOT_DIR / "cell_clusters.csv"
DEFAULT_OUTPUT_PICKLE_PATH = ROOT_DIR / "cell_clusters.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "clustering"
DEFAULT_FIGURE_FORMAT = "svg"

CONDUCTANCE_COLS = ["gNa", "gKCa", "gKd", "gCaT", "gA", "gL", "gH", "gImi", "gCaS"]
CATEGORICAL_ANNOTATION_COLS = ["silencing_status", "any_bursting_before_silence"]


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_feature_columns(table: pd.DataFrame, feature_set: str) -> list:
    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    if feature_set == "conductance":
        return [c for c in CONDUCTANCE_COLS if c in numeric_cols]
    if feature_set == "behavioral":
        return [c for c in numeric_cols if c not in CONDUCTANCE_COLS]
    if feature_set == "all":
        return numeric_cols
    raise ValueError(f"unknown feature_set: {feature_set}")


def drop_zero_variance(numeric: pd.DataFrame, std_eps: float = 1e-8) -> tuple:
    """Drops columns whose std is ~0 -- uninformative for clustering and
    would divide-by-~0 under z-scoring. This is a numerical-degeneracy
    check, not a "low variance contribution" cut (see module docstring for
    why the latter is deliberately NOT done).
    """
    std = numeric.std(ddof=0)
    keep = std[std > std_eps].index.tolist()
    dropped = std[std <= std_eps].index.tolist()
    return numeric[keep], dropped


def report_collinear_pairs(numeric: pd.DataFrame, corr_thresh: float = 0.9) -> list:
    """Flags (does not drop) feature pairs whose Pearson correlation
    exceeds corr_thresh, over whatever complete-case rows the caller
    passes in. Purely diagnostic -- see module docstring for why the
    raw-vs-pca clustering comparison, not manual column dropping, is the
    intended response to collinearity here.
    """
    corr = numeric.corr()
    pairs = []
    for a, b in combinations(numeric.columns, 2):
        r = corr.loc[a, b]
        if np.isfinite(r) and abs(r) >= corr_thresh:
            pairs.append((a, b, float(r)))
    return sorted(pairs, key=lambda t: -abs(t[2]))


def prune_correlated_features(numeric: pd.DataFrame, corr_thresh: float = 0.9) -> tuple:
    """Drops one column from each highly-correlated pair (|r| >= corr_thresh)
    before clustering -- promoted from "flag only" after confirming directly
    this matters: with the full 69-cell dataset, 3 features (baseline_freq_hz,
    peak_test_freq_hz, max_rebound_spike_count, mutually r=0.94-0.99) all
    measure essentially the same "how excitable is this cell" axis, but a
    raw standardized-Euclidean clustering counts each one as an independent
    dimension -- silently giving that one underlying axis ~3x the weight of
    any other, genuinely-independent feature. This is very likely why
    unconstrained silhouette-based k selection kept climbing all the way to
    the k_range boundary (k=7 of a max-8 search) instead of settling
    anywhere -- more clusters can keep "improving" apparent separation when
    a few over-weighted, redundant dimensions dominate the distance metric,
    which isn't the same thing as real structure (silhouette here never
    exceeded ~0.34, well under the ~0.5 rule-of-thumb for genuine
    clustering, at ANY k).

    Processes pairs sorted by |r| descending (the strongest redundancies
    first) and greedily drops the SECOND column of each pair not already
    dropped -- deterministic, and avoids double-dropping a column that
    appears in more than one flagged pair (e.g. all three of the mutually-
    correlated features above only cost two drops, not three, since once
    peak_test_freq_hz and max_rebound_spike_count are both gone from the
    baseline_freq_hz pairs there's nothing left to drop there).
    """
    pairs = report_collinear_pairs(numeric, corr_thresh)
    dropped = []
    for a, b, _r in pairs:
        if a in dropped or b in dropped:
            continue
        dropped.append(b)
    kept_cols = [c for c in numeric.columns if c not in dropped]
    return numeric[kept_cols], dropped, pairs


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _best_k_by_silhouette(X: np.ndarray, k_range) -> dict:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        if k >= len(X):
            continue
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X, labels)
    if not scores:
        return {"k": None, "scores": {}}
    best_k = max(scores, key=scores.get)
    return {"k": best_k, "scores": scores}


def cluster_view(X: np.ndarray, k: int) -> dict:
    """Fits hierarchical (ward), k-means, and Gaussian-mixture clustering
    at a shared k on the same standardized matrix X, so their label sets
    are directly comparable via adjusted Rand index -- a real cluster
    boundary should be recovered by more than one method; one that only
    one method finds is a weaker claim.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score

    Z = linkage(X, method="ward")
    hier_labels = fcluster(Z, t=k, criterion="maxclust")
    kmeans_labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    gmm_labels = GaussianMixture(n_components=k, random_state=0).fit_predict(X)

    agreement = {
        "hier_vs_kmeans": adjusted_rand_score(hier_labels, kmeans_labels),
        "hier_vs_gmm": adjusted_rand_score(hier_labels, gmm_labels),
        "kmeans_vs_gmm": adjusted_rand_score(kmeans_labels, gmm_labels),
    }
    return {"linkage": Z, "hier_labels": hier_labels, "kmeans_labels": kmeans_labels,
            "gmm_labels": gmm_labels, "agreement": agreement}


def run_clustering(table: pd.DataFrame, feature_set: str = "behavioral",
                   min_feature_coverage: float = 0.85, corr_thresh: float = 0.9,
                   k: int = None, k_range=range(2, 9)) -> dict:
    cols = select_feature_columns(table, feature_set)
    numeric = table[cols]
    numeric, zero_var_dropped = drop_zero_variance(numeric)
    numeric, collinear_dropped, collinear_pairs = prune_correlated_features(numeric, corr_thresh)
    complete, kept_cols, coverage_dropped, excluded = select_complete_case_matrix(
        numeric, min_feature_coverage)

    if len(complete) < 4:
        return {"status": "insufficient_cells", "n_complete": len(complete),
                "excluded_cell_ids": excluded,
                "dropped_feature_names": zero_var_dropped + collinear_dropped + coverage_dropped}

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_raw = scaler.fit_transform(complete.values)

    pca = PCA()
    scores = pca.fit_transform(X_raw)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cum_var, 0.90) + 1)
    n_pcs = max(2, min(n_pcs, scores.shape[1]))
    X_pca = scores[:, :n_pcs]

    chosen_k = k
    silhouette_raw = _best_k_by_silhouette(X_raw, k_range)
    if chosen_k is None:
        chosen_k = silhouette_raw["k"]
    if chosen_k is None:
        return {"status": "no_valid_k", "n_complete": len(complete),
                "excluded_cell_ids": excluded,
                "dropped_feature_names": zero_var_dropped + collinear_dropped + coverage_dropped}

    raw_result = cluster_view(X_raw, chosen_k)
    pca_result = cluster_view(X_pca, chosen_k)

    from sklearn.metrics import adjusted_rand_score
    cross_view_agreement = adjusted_rand_score(raw_result["hier_labels"], pca_result["hier_labels"])

    return {
        "status": "ok", "feature_set": feature_set, "cell_ids": complete.index.tolist(),
        "excluded_cell_ids": excluded,
        "dropped_feature_names": zero_var_dropped + collinear_dropped + coverage_dropped,
        "feature_names": kept_cols, "collinear_pairs": collinear_pairs,
        "k": chosen_k, "silhouette_by_k": silhouette_raw["scores"],
        "X_raw": X_raw, "X_pca": X_pca, "n_pcs_90pct": n_pcs,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "raw": raw_result, "pca": pca_result, "cross_view_agreement": cross_view_agreement,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_dendrogram(result: dict, outdir: Path, command: str, fig_format: str) -> None:
    from scipy.cluster.hierarchy import dendrogram

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(result["cell_ids"])), 5))
    dendrogram(result["raw"]["linkage"], labels=result["cell_ids"], ax=ax, leaf_font_size=6,
              color_threshold=result["raw"]["linkage"][-(result["k"] - 1), 2] if result["k"] > 1 else None)
    ax.set_title(f"Hierarchical clustering ({result['feature_set']} features, "
                f"k={result['k']}, n={len(result['cell_ids'])})", fontsize=10)
    ax.set_ylabel("Ward distance")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.text(0.5, 0.005, command, ha="center", va="bottom", fontsize=5, color="gray")
    fig.savefig(outdir / f"dendrogram_{result['feature_set']}.{fig_format}")
    plt.close(fig)


def plot_cluster_pca_scatter(result: dict, outdir: Path, command: str, fig_format: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    scores = result["X_pca"]
    labels = result["raw"]["hier_labels"]
    var = result["pca_explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        ax.scatter(scores[mask, 0], scores[mask, 1], s=35, color=cmap((lbl - 1) % 10), label=f"cluster {lbl}")
    for i, cid in enumerate(result["cell_ids"]):
        ax.annotate(cid, (scores[i, 0], scores[i, 1]), fontsize=5, alpha=0.7)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title(f"Clusters on PCA scores ({result['feature_set']} features, "
                f"colored by raw-space hierarchical cluster)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.text(0.5, 0.005, command, ha="center", va="bottom", fontsize=5, color="gray")
    fig.savefig(outdir / f"pca_scatter_{result['feature_set']}.{fig_format}")
    plt.close(fig)


def plot_cluster_feature_means(result: dict, table: pd.DataFrame, outdir: Path,
                               command: str, fig_format: str) -> None:
    """Per-cluster mean of each z-scored feature, as a heatmap -- the
    fastest way to see what actually distinguishes each cluster (large
    positive/negative cells) without eyeballing raw units.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    labels = result["raw"]["hier_labels"]
    X = result["X_raw"]
    feature_names = result["feature_names"]
    df = pd.DataFrame(X, index=result["cell_ids"], columns=feature_names)
    df["cluster"] = labels
    means = df.groupby("cluster").mean()

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(feature_names)), max(3, 0.6 * len(means))))
    # interpolation="nearest" (2026-08-12, per explicit instruction): this
    # is a discrete cluster x feature matrix, not a spatial grid -- bicubic
    # blending between adjacent rows/columns fabricated values between
    # unrelated clusters/features that don't lie on any real continuum.
    im = ax.imshow(means.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([f"cluster {c}" for c in means.index], fontsize=8)
    fig.colorbar(im, ax=ax, label="mean z-score")
    ax.set_title(f"Per-cluster feature means ({result['feature_set']} features)", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.text(0.5, 0.005, command, ha="center", va="bottom", fontsize=5, color="gray")
    fig.savefig(outdir / f"cluster_feature_means_{result['feature_set']}.{fig_format}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-csv", default=DEFAULT_MASTER_CSV_PATH)
    parser.add_argument("--feature-set", default="behavioral", choices=["behavioral", "conductance", "all"])
    parser.add_argument("--min-feature-coverage", type=float, default=0.85)
    parser.add_argument("--corr-thresh", type=float, default=0.9,
                        help="Feature pairs correlated at or above this (absolute value) get one "
                             "column pruned before clustering -- see prune_correlated_features.")
    parser.add_argument("--k", type=int, default=3,
                        help="Cluster count. Defaults to 3 (the known AB/PD, LP, PY count) rather "
                             "than unconstrained silhouette-based selection -- confirmed directly "
                             "silhouette alone climbs to the k-range boundary on this data without "
                             "ever exceeding weak (~0.34) scores, i.e. it doesn't reliably pick a "
                             "real k here. Pass e.g. --k 0 to fall back to automatic selection over "
                             "--k-min..--k-max for exploration.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument("--output-pickle", default=DEFAULT_OUTPUT_PICKLE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)

    table = pd.read_csv(args.master_csv, index_col="cell_id")
    k = None if args.k == 0 else args.k  # 0 == request automatic silhouette-based selection
    result = run_clustering(table, feature_set=args.feature_set,
                            min_feature_coverage=args.min_feature_coverage,
                            corr_thresh=args.corr_thresh, k=k,
                            k_range=range(args.k_min, args.k_max + 1))

    if result["status"] != "ok":
        print(f"Clustering skipped: {result['status']} (n_complete={result.get('n_complete')})")
        return

    print(f"Clustering {len(result['cell_ids'])} cells on {len(result['feature_names'])} "
         f"'{args.feature_set}' features (k={result['k']})")
    if result["excluded_cell_ids"]:
        print(f"Excluded for missing data: {result['excluded_cell_ids']}")
    if result["dropped_feature_names"]:
        print(f"Dropped features (zero-variance, collinear, or low coverage): {result['dropped_feature_names']}")
    if result["collinear_pairs"]:
        print(f"\nCollinear pairs found (|r| >= {args.corr_thresh}) -- one column of each pruned:")
        for a, b, r in result["collinear_pairs"]:
            print(f"  {a} <-> {b}: r={r:+.3f}")

    print(f"\nSilhouette score by k: "
         + ", ".join(f"{kk}={vv:.3f}" for kk, vv in sorted(result["silhouette_by_k"].items())))
    print(f"n PCs for 90% variance: {result['n_pcs_90pct']}")

    print("\nMethod agreement (adjusted Rand index, raw feature space):")
    for pair, score in result["raw"]["agreement"].items():
        print(f"  {pair}: {score:.3f}")
    print(f"Raw-space vs PCA-space hierarchical clustering agreement: {result['cross_view_agreement']:.3f}")

    cluster_df = pd.DataFrame({
        "cell_id": result["cell_ids"],
        "cluster_hier_raw": result["raw"]["hier_labels"],
        "cluster_kmeans_raw": result["raw"]["kmeans_labels"],
        "cluster_gmm_raw": result["raw"]["gmm_labels"],
        "cluster_hier_pca": result["pca"]["hier_labels"],
    }).set_index("cell_id")
    for col in CATEGORICAL_ANNOTATION_COLS:
        if col in table.columns:
            cluster_df[col] = table.loc[cluster_df.index, col]
    cluster_df.to_csv(args.output_csv)
    print(f"\nCluster assignments written to {args.output_csv}")

    with open(args.output_pickle, "wb") as f:
        pickle.dump(result, f)
    print(f"Full clustering result written to {args.output_pickle}")

    if not args.no_plot:
        figdir = Path(args.figures_dir)
        plot_dendrogram(result, figdir, command, args.figure_format)
        plot_cluster_pca_scatter(result, figdir, command, args.figure_format)
        plot_cluster_feature_means(result, table, figdir, command, args.figure_format)
        print(f"Figures written to {figdir}/")


if __name__ == "__main__":
    main()
