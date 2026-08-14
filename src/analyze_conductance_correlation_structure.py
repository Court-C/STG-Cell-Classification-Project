"""Part B: does conductance correlation STRUCTURE (not values) distinguish
cell groups? Literature finding (Ghosh, Nunez-Vargas et al., "Conductance
Ratios and Cellular Identity"): each STG cell type shows its own distinct
pattern of PAIRWISE LINEAR CORRELATIONS between conductances (gA vs gH,
gCaT vs gKCa, etc.) -- a fundamentally different clustering objective than
anything tried so far in this project, which has always clustered on
feature VALUES via Euclidean distance, never on within-group correlation
structure itself.

No new simulation -- reads track_a_features.csv (9 conductances x 69
cells) directly.

Three things, in order:
  1. Global pairwise correlation matrix (heatmap, reference baseline).
  2. Test whether the existing, best-trusted behavioral k=3 clustering
     already shows tighter within-group correlation structure than chance
     (a direct, independent check on a clustering derived a completely
     different way).
  3. A from-scratch local-search partition optimizer whose OBJECTIVE is
     within-group correlation sharpness directly (not point-distance),
     validated against a null-permutation baseline built the same way as
     select_clustering_features.py's -- same discipline, new objective.

Score definition: for a candidate partition, mean |r| across all 36
conductance pairs, computed within each group (groups below
MIN_GROUP_SIZE excluded -- a 2-3 cell group can trivially show near-
perfect correlation by chance, same reasoning as MIN_CLUSTER_SIZE in
select_clustering_features.py), then averaged across groups weighted by
group size.
"""

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

DEFAULT_TRACK_A_PATH = ROOT_DIR / "track_a_features.csv"
DEFAULT_MASTER_CSV_PATH = ROOT_DIR / "master_features.csv"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "conductance_correlation_structure"
DEFAULT_FIGURE_FORMAT = "png"

CONDUCTANCE_COLS = ["gNa", "gKCa", "gKd", "gCaT", "gA", "gL", "gH", "gImi", "gCaS"]
MIN_GROUP_SIZE = 5
DEFAULT_RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def mean_abs_correlation(df: pd.DataFrame, cols=CONDUCTANCE_COLS) -> float:
    corr = df[cols].corr()
    vals = [abs(corr.loc[a, b]) for a, b in combinations(cols, 2) if np.isfinite(corr.loc[a, b])]
    return float(np.mean(vals)) if vals else float("nan")


def score_partition(df: pd.DataFrame, labels: np.ndarray, cols=CONDUCTANCE_COLS,
                    min_group_size: int = MIN_GROUP_SIZE) -> float:
    total, total_weight = 0.0, 0
    for lbl in np.unique(labels):
        mask = labels == lbl
        n = int(mask.sum())
        if n < min_group_size:
            continue
        s = mean_abs_correlation(df[mask], cols)
        if np.isfinite(s):
            total += s * n
            total_weight += n
    return total / total_weight if total_weight > 0 else float("-inf")


# ---------------------------------------------------------------------------
# Part B.1/B.2: global heatmap + existing-clustering test
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, outpath: Path, cols=CONDUCTANCE_COLS,
                             title: str = "Global conductance correlation matrix") -> None:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # interpolation="nearest" (2026-08-12, per explicit instruction): this
    # is a discrete conductance x conductance correlation matrix, not a
    # spatial grid -- bicubic blending fabricated values between unrelated
    # conductance pairs, directly contradicting the exact r-values already
    # printed on each cell below.
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def test_existing_clustering(df: pd.DataFrame, labels: pd.Series, cols=CONDUCTANCE_COLS,
                             min_group_size: int = MIN_GROUP_SIZE, n_null: int = 200,
                             random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    common = df.index.intersection(labels.index)
    df_c, labels_c = df.loc[common], labels.loc[common].to_numpy()

    real_score = score_partition(df_c, labels_c, cols, min_group_size)
    group_sizes = pd.Series(labels_c).value_counts().to_dict()

    rng = np.random.default_rng(random_state)
    null_scores = []
    for _ in range(n_null):
        shuffled = rng.permutation(labels_c)
        null_scores.append(score_partition(df_c, shuffled, cols, min_group_size))
    null_scores = np.array(null_scores)
    finite = null_scores[np.isfinite(null_scores)]
    percentile = float(100.0 * np.mean(finite <= real_score)) if len(finite) else None

    return {"n_cells": len(common), "group_sizes": group_sizes, "real_score": real_score,
           "null_scores": null_scores, "percentile": percentile}


# ---------------------------------------------------------------------------
# Part B.3: from-scratch local-search partition optimizer
# ---------------------------------------------------------------------------

def local_search_partition(df: pd.DataFrame, k: int, cols=CONDUCTANCE_COLS,
                           min_group_size: int = MIN_GROUP_SIZE, n_restarts: int = 3,
                           seed_labels: np.ndarray = None, random_state: int = DEFAULT_RANDOM_STATE,
                           max_sweeps: int = 25) -> tuple:
    """Hill-climbing: for each cell, in random order, move it to whichever
    group most improves the total score (never letting a move shrink its
    source group below min_group_size), repeat until a full sweep produces
    no improving move (local optimum) or max_sweeps is hit. Multiple
    restarts (a seed partition, e.g. the existing clustering, plus random
    inits) reduce dependence on the starting point; returns the best
    (labels, score) found across restarts.
    """
    rng = np.random.default_rng(random_state)
    n = len(df)
    starts = []
    if seed_labels is not None:
        starts.append(np.asarray(seed_labels).copy())
    for _ in range(n_restarts):
        starts.append(rng.integers(0, k, size=n))

    best_labels, best_score = None, float("-inf")
    for start in starts:
        labels = start.copy()
        for _sweep in range(max_sweeps):
            improved = False
            for i in rng.permutation(n):
                current_lbl = labels[i]
                current_score = score_partition(df, labels, cols, min_group_size)
                best_local_lbl, best_local_score = current_lbl, current_score
                for cand_lbl in range(k):
                    if cand_lbl == current_lbl:
                        continue
                    if np.sum(labels == current_lbl) - 1 < min_group_size:
                        continue
                    trial = labels.copy()
                    trial[i] = cand_lbl
                    trial_score = score_partition(df, trial, cols, min_group_size)
                    if trial_score > best_local_score:
                        best_local_score, best_local_lbl = trial_score, cand_lbl
                if best_local_lbl != current_lbl:
                    labels[i] = best_local_lbl
                    improved = True
            if not improved:
                break
        final_score = score_partition(df, labels, cols, min_group_size)
        if final_score > best_score:
            best_score, best_labels = final_score, labels.copy()

    return best_labels, best_score


def search_with_null(df: pd.DataFrame, k: int, cols=CONDUCTANCE_COLS,
                     min_group_size: int = MIN_GROUP_SIZE, n_restarts: int = 3,
                     seed_labels: np.ndarray = None, n_null: int = 50,
                     random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    labels, real_score = local_search_partition(df, k, cols, min_group_size, n_restarts,
                                                seed_labels, random_state)

    rng = np.random.default_rng(random_state + 1)
    null_scores = []
    for i in range(n_null):
        shuffled = df.copy()
        for c in cols:
            shuffled[c] = rng.permutation(shuffled[c].values)
        _, null_score = local_search_partition(shuffled, k, cols, min_group_size, n_restarts,
                                               seed_labels=None, random_state=random_state + 100 + i)
        null_scores.append(null_score)
    null_scores = np.array(null_scores)
    finite = null_scores[np.isfinite(null_scores)]
    percentile = float(100.0 * np.mean(finite <= real_score)) if len(finite) else None

    return {"k": k, "labels": labels, "real_score": real_score, "null_scores": null_scores,
           "percentile": percentile,
           "group_sizes": pd.Series(labels).value_counts().sort_index().to_dict()}


# ---------------------------------------------------------------------------
# Bootstrap stability of the winning partition
# ---------------------------------------------------------------------------

def bootstrap_stability(df: pd.DataFrame, reference_labels: pd.Series, k: int, cols=CONDUCTANCE_COLS,
                        min_group_size: int = MIN_GROUP_SIZE, n_restarts: int = 3,
                        n_bootstrap: int = 100, random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Resamples cells with replacement, reruns the from-scratch local
    search on each resample, and compares the resulting partition to the
    reference (full, non-resampled) partition via adjusted Rand index.
    Duplicate rows within a resample (the same cell drawn more than once)
    are collapsed to one label per cell via majority vote before scoring.
    A partition that's just a fragile local optimum specific to this one
    69-cell draw should show low agreement across resamples; a real,
    robust partition should keep reappearing.
    """
    from collections import Counter
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(random_state)
    cell_ids = df.index.to_numpy()
    n = len(cell_ids)
    ari_scores = []

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        resampled = df.iloc[sample_idx]
        labels, _ = local_search_partition(resampled, k, cols, min_group_size, n_restarts,
                                           seed_labels=None, random_state=random_state + 1000 + b)
        sampled_ids = cell_ids[sample_idx]
        per_cell_labels: dict = {}
        for cid, lbl in zip(sampled_ids, labels):
            per_cell_labels.setdefault(cid, []).append(int(lbl))
        collapsed = {cid: Counter(lbls).most_common(1)[0][0] for cid, lbls in per_cell_labels.items()}

        common = [cid for cid in collapsed if cid in reference_labels.index]
        if len(common) < min_group_size * k:
            continue
        boot_lbls = [collapsed[cid] for cid in common]
        ref_lbls = [reference_labels.loc[cid] for cid in common]
        ari_scores.append(adjusted_rand_score(ref_lbls, boot_lbls))

    ari_scores = np.array(ari_scores)
    finite = ari_scores[np.isfinite(ari_scores)]
    return {"n_bootstrap": n_bootstrap, "n_valid": len(finite), "ari_scores": ari_scores,
           "ari_mean": float(finite.mean()) if len(finite) else None,
           "ari_median": float(np.median(finite)) if len(finite) else None,
           "frac_above_0p5": float(np.mean(finite > 0.5)) if len(finite) else None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track-a-csv", default=DEFAULT_TRACK_A_PATH)
    parser.add_argument("--master-csv", default=DEFAULT_MASTER_CSV_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT)
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    parser.add_argument("--n-null-existing", type=int, default=200)
    parser.add_argument("--n-null-search", type=int, default=50)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip Part B.3 (from-scratch search) -- just B.1/B.2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figdir = Path(args.figures_dir)
    df = pd.read_csv(args.track_a_csv, index_col="cell_id")

    print(f"Loaded {len(df)} cells x {len(CONDUCTANCE_COLS)} conductances")
    plot_correlation_heatmap(df, figdir / f"global_correlation_heatmap.{args.figure_format}")
    global_score = mean_abs_correlation(df)
    print(f"Global mean |r| across all 36 conductance pairs (whole population): {global_score:.3f}")
    print(f"Figure written to {figdir}/global_correlation_heatmap.{args.figure_format}")

    print("\n" + "=" * 70 + "\nPart B.2: existing behavioral clustering vs. null\n" + "=" * 70)
    master = pd.read_csv(args.master_csv, index_col="cell_id")
    if "cluster_hier_raw" in master.columns:
        labels = master["cluster_hier_raw"].dropna()
    else:
        cell_clusters_path = ROOT_DIR / "cell_clusters.csv"
        labels = pd.read_csv(cell_clusters_path, index_col="cell_id")["cluster_hier_raw"]
    result = test_existing_clustering(df, labels, min_group_size=args.min_group_size,
                                      n_null=args.n_null_existing, random_state=args.random_state)
    print(f"n cells: {result['n_cells']}, group sizes: {result['group_sizes']}")
    print(f"Real within-group mean |r|: {result['real_score']:.3f}")
    print(f"Null distribution (n={args.n_null_existing}): "
         f"mean={result['null_scores'].mean():.3f}, std={result['null_scores'].std():.3f}")
    print(f"Real score percentile vs. null: {result['percentile']:.1f}")

    if args.skip_search:
        return

    print("\n" + "=" * 70 + "\nPart B.3: from-scratch correlation-clustering search\n" + "=" * 70)
    for k in range(args.k_min, args.k_max + 1):
        seed = None
        if k == 3 and "cluster_hier_raw" in master.columns:
            common = df.index.intersection(labels.index)
            seed_map = {cid: i for i, cid in enumerate(sorted(labels.unique()))}
            seed = np.array([seed_map[labels.loc[cid]] for cid in df.index if cid in common])
            seed = None if len(seed) != len(df) else seed
        res = search_with_null(df, k, min_group_size=args.min_group_size, n_restarts=args.n_restarts,
                               seed_labels=seed, n_null=args.n_null_search, random_state=args.random_state)
        print(f"\nk={k}: best score={res['real_score']:.3f}, group sizes={res['group_sizes']}")
        print(f"  null (n={args.n_null_search}): mean={res['null_scores'].mean():.3f}, "
             f"std={res['null_scores'].std():.3f}, percentile={res['percentile']:.1f}")


if __name__ == "__main__":
    main()
