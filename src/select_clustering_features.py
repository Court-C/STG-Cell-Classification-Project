"""Feature-selection search: which subset of measurable parameters makes
k=3 the natural silhouette/GMM-BIC winner, rather than something we force
by policy (see cluster_features.py's --k default).

This is a hypothesis test, not a guaranteed-success search: the hypothesis
is that some combination of conductance/behavioral parameters separates
these cells into 3 groups distinctly enough to serve as a real classifier
for future (unlabeled) cells. That may or may not be true. Both outcomes
are reported plainly -- see main()'s summary and REPORT_NOT_SUPPORTED_MSG.

No ground-truth per-cell type label exists anywhere in this repo (checked
directly), so this whole search is unsupervised: there is no labeled fold
to validate a chosen subset against. Two safeguards compensate for the
overfitting risk that comes with searching ~24 candidate features over
~67-69 cells with no label to check against:

  - Null-permutation baseline: rerun the exact same greedy search on
    independently column-shuffled data (same marginal distributions, no
    real cross-feature structure) many times, and report where the real
    search's result falls against that chance distribution. If a search
    over pure noise regularly finds something "as good", the real result
    isn't trustworthy.
  - Bootstrap stability: resample cells with replacement on the FIXED
    winning subset (no re-search) and confirm k=3 keeps winning across
    resamples, not just on this one specific draw of cells.

A third, independent check -- a Bayesian (Dirichlet-process) Gaussian
mixture, which infers its own component count with a built-in preference
for fewer rather than us manually sweeping k -- is run on the winning
subset as well. Agreement between that self-selected k and the manually
swept silhouette/BIC k=3 result is separate evidence, not derived from the
same k=2..8 sweep the rest of this script uses.

Candidates are drawn from master_features.csv. Two pools are meaningful:
  - "full": conductances + behavioral features. Conductances are, in
    principle, measurable post-hoc in a real cell (ion channel expression
    quantification), so they're fair game -- this establishes the ceiling.
  - "behavioral": behavioral features only. This is the practically
    deployable pool for a real recorded cell where conductances aren't
    known -- run this as a second stage and compare against "full" (see
    --stage both, the default).

Run via: python select_clustering_features.py --stage both
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from consolidate_features import select_complete_case_matrix, DEFAULT_OUTPUT_CSV_PATH as DEFAULT_MASTER_CSV_PATH
from cluster_features import CONDUCTANCE_COLS, drop_zero_variance

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "clustering_search"
DEFAULT_FIGURE_FORMAT = "svg"

DEFAULT_TARGET_K = 3
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 8
DEFAULT_MAX_SUBSET_SIZE = 8
DEFAULT_MIN_SUBSET_SIZE = 3
DEFAULT_MIN_FEATURE_COVERAGE = 0.85
DEFAULT_N_NULL = 100
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_RANDOM_STATE = 0
GMM_N_INIT_SEARCH = 1
GMM_N_INIT_FINAL = 5
BGM_N_INIT = 3

NOT_SUPPORTED_MSG = (
    "No feature subset cleared both safeguards -- at least with the "
    "features currently measured, these cells do not separate into 3 "
    "distinguishable groups strongly enough to trust. This is a valid "
    "result, not a failed search."
)


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

DEFAULT_MIN_UNIQUE_VALUES = 10


def build_candidate_pool(table: pd.DataFrame, pool: str, min_feature_coverage: float,
                         min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES) -> dict:
    """Fixes the candidate column list and the complete-case row set ONCE,
    from the full pool, so that as the greedy search evaluates different
    subsets, N (cell count) stays constant across evaluations -- otherwise
    silhouette/BIC scores for different subsets wouldn't be comparable.

    Also drops low-cardinality features (fewer than min_unique_values
    distinct values) BEFORE the search ever sees them. Caught directly
    during development: n_burst_transitions has only 3 distinct integer
    values across 67 cells (30/36/3) -- used alone it produces a
    trivially "perfect" silhouette (1.000 at k=3) because it's already a
    discrete binning, not a continuous multivariate signal, and its
    artificially huge score dominated every later round of the greedy
    search regardless of the min_subset_size floor. Every other candidate
    column has 38+ distinct values, comfortably above this threshold, so
    the filter is targeted at this specific pathology, not a broad cut.
    """
    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    if pool == "full":
        cols = numeric_cols
    elif pool == "behavioral":
        cols = [c for c in numeric_cols if c not in CONDUCTANCE_COLS]
    else:
        raise ValueError(f"unknown pool: {pool}")

    numeric = table[cols]
    numeric, zero_var_dropped = drop_zero_variance(numeric)
    complete, kept_cols, coverage_dropped, excluded = select_complete_case_matrix(
        numeric, min_feature_coverage)

    nunique = complete[kept_cols].nunique()
    low_cardinality_dropped = nunique[nunique < min_unique_values].index.tolist()
    kept_cols = [c for c in kept_cols if c not in low_cardinality_dropped]
    complete = complete[kept_cols]

    return {
        "pool": pool, "complete": complete, "candidate_cols": kept_cols,
        "zero_var_dropped": zero_var_dropped, "coverage_dropped": coverage_dropped,
        "low_cardinality_dropped": low_cardinality_dropped, "excluded_cell_ids": excluded,
    }


# ---------------------------------------------------------------------------
# Objective: does target_k win on silhouette AND GMM BIC?
# ---------------------------------------------------------------------------

MIN_CLUSTER_SIZE = 3


def compute_k_curves(X: np.ndarray, k_range, gmm_n_init: int, random_state: int,
                     min_cluster_size: int = MIN_CLUSTER_SIZE) -> dict:
    """Skips any k where hierarchical clustering doesn't actually produce k
    well-populated clusters -- caught directly during development: a single
    low-cardinality integer feature (n_burst_transitions, effectively ~3-4
    distinct values across 67 cells) let Ward clustering "collapse" onto
    fewer real groups than requested for several k, producing degenerate
    near-empty clusters that inflate silhouette artificially (silhouette=
    1.000 exactly at k=3, sklearn warning "Number of distinct clusters
    found smaller than n_clusters"). That isn't real multivariate
    structure -- it's binning on one discrete feature -- so any k where a
    cluster has fewer than min_cluster_size members is treated as
    infeasible for that k, the same as k >= n.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    n = X.shape[0]
    curves = {}
    for k in k_range:
        if k >= n:
            continue
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        counts = np.bincount(labels)
        if len(counts) < k or counts.min() < min_cluster_size:
            continue
        sil = silhouette_score(X, labels)
        gmm = GaussianMixture(n_components=k, n_init=gmm_n_init, random_state=random_state).fit(X)
        curves[k] = {"silhouette": float(sil), "bic": float(gmm.bic(X))}
    return curves


def score_subset(X: np.ndarray, target_k: int, k_range, gmm_n_init: int, random_state: int):
    """Returns None if target_k or every comparison k is infeasible.
    Otherwise a dict with the full curves plus silhouette_margin (higher is
    better, target_k's silhouette minus the best of the rest) and
    bic_margin (higher is better, the best of the rest's BIC minus
    target_k's BIC -- BIC is lower-is-better so this sign makes both
    margins "positive means target_k wins").
    """
    curves = compute_k_curves(X, k_range, gmm_n_init, random_state)
    if target_k not in curves:
        return None
    others = {k: v for k, v in curves.items() if k != target_k}
    if not others:
        return None
    sil_margin = curves[target_k]["silhouette"] - max(v["silhouette"] for v in others.values())
    bic_margin = min(v["bic"] for v in others.values()) - curves[target_k]["bic"]
    return {"curves": curves, "silhouette_margin": sil_margin, "bic_margin": bic_margin}


def combined_score(result: dict, n_samples: int) -> float:
    """Smooth scalar used to drive greedy selection: silhouette_margin is
    already O(0.01-1), bic_margin is normalized by n_samples to bring it to
    a comparable scale (BIC's magnitude scales with n via its -2*loglik
    term). This is a search-guidance heuristic, not the final pass/fail
    criterion -- that's the hard gate (both margins > 0) applied to the
    search's end result, see gate_passed below.
    """
    return result["silhouette_margin"] + result["bic_margin"] / n_samples


def gate_passed(result: dict) -> bool:
    return result["silhouette_margin"] > 0 and result["bic_margin"] > 0


# ---------------------------------------------------------------------------
# Greedy forward selection
# ---------------------------------------------------------------------------

def greedy_forward_select(complete: pd.DataFrame, candidate_cols: list, target_k: int,
                          k_range, max_subset_size: int, gmm_n_init: int, random_state: int,
                          min_subset_size: int = DEFAULT_MIN_SUBSET_SIZE) -> dict:
    """Greedy forward selection. Caught directly during development: a
    single low-cardinality integer feature (n_burst_transitions, ~3-4
    distinct values across 67 cells) can produce a trivially "perfect"
    silhouette (1.000) by discretizing cells into near-zero-spread bins --
    that's binning on one feature, not real multivariate structure, and
    since nothing can beat a perfect score, greedy would stop at size 1 and
    never even look for a genuine multi-feature combination. So the
    "stop if no candidate improves the score" rule is only allowed to
    fire once min_subset_size features are already selected -- before
    that, keep adding whichever candidate scores best each round
    regardless of whether it improves on the previous round.
    """
    from sklearn.preprocessing import StandardScaler

    n = len(complete)
    selected = []
    remaining = list(candidate_cols)
    history = []
    best_gated = None  # (subset, result, combined) -- best gated point seen along the path

    def evaluate(cols):
        X = StandardScaler().fit_transform(complete[cols].values)
        return score_subset(X, target_k, k_range, gmm_n_init, random_state)

    current_combined = -np.inf
    while remaining and len(selected) < max_subset_size:
        round_results = []
        for c in remaining:
            result = evaluate(selected + [c])
            if result is None:
                continue
            round_results.append((c, result, combined_score(result, n)))
        if not round_results:
            break
        round_results.sort(key=lambda t: t[2], reverse=True)
        best_c, best_result, best_combined = round_results[0]

        if len(selected) >= min_subset_size and best_combined <= current_combined:
            break  # no candidate improves on the current subset -- stop

        selected.append(best_c)
        remaining.remove(best_c)
        current_combined = best_combined
        passed = gate_passed(best_result)
        history.append({
            "added": best_c, "subset": list(selected), "combined_score": best_combined,
            "silhouette_margin": best_result["silhouette_margin"],
            "bic_margin": best_result["bic_margin"], "gate_passed": passed,
            "curves": best_result["curves"],
        })
        if passed and len(selected) >= min_subset_size and (best_gated is None or best_combined > best_gated[2]):
            best_gated = (list(selected), best_result, best_combined)

    final_result = evaluate(selected) if selected else None
    return {
        "history": history,
        "final_subset": list(selected),
        "final_result": final_result,
        "final_combined_score": current_combined if selected else None,
        "best_gated_subset": best_gated[0] if best_gated else None,
        "best_gated_result": best_gated[1] if best_gated else None,
        "best_gated_combined_score": best_gated[2] if best_gated else None,
    }


# ---------------------------------------------------------------------------
# Safeguard 1: null-permutation baseline
# ---------------------------------------------------------------------------

def _one_null_search(seed: int, complete: pd.DataFrame, candidate_cols: list, target_k: int,
                     k_range, max_subset_size: int, gmm_n_init: int, random_state: int,
                     min_subset_size: int) -> float:
    rng = np.random.default_rng(seed)
    shuffled = complete[candidate_cols].copy()
    for col in candidate_cols:
        shuffled[col] = rng.permutation(shuffled[col].values)
    result = greedy_forward_select(shuffled, candidate_cols, target_k, k_range,
                                   max_subset_size, gmm_n_init, random_state, min_subset_size)
    best = result["best_gated_combined_score"]
    return best if best is not None else -np.inf


def null_permutation_test(complete: pd.DataFrame, candidate_cols: list, real_best_combined: float,
                          target_k: int, k_range, max_subset_size: int, gmm_n_init: int,
                          random_state: int, n_null: int, jobs: int,
                          min_subset_size: int = DEFAULT_MIN_SUBSET_SIZE) -> dict:
    from joblib import Parallel, delayed

    null_scores = Parallel(n_jobs=jobs)(
        delayed(_one_null_search)(seed, complete, candidate_cols, target_k, k_range,
                                  max_subset_size, gmm_n_init, random_state, min_subset_size)
        for seed in range(n_null)
    )
    null_scores = np.array(null_scores, dtype=float)
    finite = null_scores[np.isfinite(null_scores)]
    if real_best_combined is None or len(finite) == 0:
        percentile = None
    else:
        percentile = float(100.0 * np.mean(finite <= real_best_combined))
    return {"null_scores": null_scores.tolist(), "n_null": n_null,
           "real_score_percentile": percentile}


# ---------------------------------------------------------------------------
# Safeguard 2: bootstrap stability of the winning subset (no re-search)
# ---------------------------------------------------------------------------

def _one_bootstrap(seed: int, complete: pd.DataFrame, subset: list, target_k: int,
                   k_range, gmm_n_init: int, random_state: int) -> bool:
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(complete), size=len(complete))
    resampled = complete.iloc[idx]
    X = StandardScaler().fit_transform(resampled[subset].values)
    result = score_subset(X, target_k, k_range, gmm_n_init, random_state)
    return result is not None and gate_passed(result)


def bootstrap_stability(complete: pd.DataFrame, subset: list, target_k: int, k_range,
                        gmm_n_init: int, random_state: int, n_bootstrap: int, jobs: int) -> dict:
    from joblib import Parallel, delayed

    outcomes = Parallel(n_jobs=jobs)(
        delayed(_one_bootstrap)(seed, complete, subset, target_k, k_range, gmm_n_init, random_state)
        for seed in range(n_bootstrap)
    )
    frac_passed = float(np.mean(outcomes)) if outcomes else 0.0
    return {"n_bootstrap": n_bootstrap, "fraction_k_wins": frac_passed}


# ---------------------------------------------------------------------------
# Safeguard 3 / independent check: Bayesian (Dirichlet-process) GMM
# ---------------------------------------------------------------------------

def bayesian_gmm_effective_k(complete: pd.DataFrame, subset: list, max_components: int,
                             random_state: int) -> dict:
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(complete[subset].values)
    n = X.shape[0]
    max_components = min(max_components, n - 1)
    weight_threshold = 1.0 / (2 * n)
    bgm = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type="dirichlet_process",
        max_iter=500, n_init=BGM_N_INIT, random_state=random_state,
    ).fit(X)
    weights = np.sort(bgm.weights_)[::-1]
    effective_k = int(np.sum(weights > weight_threshold))
    return {"effective_k": effective_k, "weights": weights.tolist(),
           "weight_threshold": weight_threshold, "max_components": max_components}


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------

def run_stage(table: pd.DataFrame, pool: str, target_k: int, k_min: int, k_max: int,
             max_subset_size: int, min_feature_coverage: float, n_null: int, n_bootstrap: int,
             jobs: int, random_state: int, min_subset_size: int = DEFAULT_MIN_SUBSET_SIZE,
             min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES) -> dict:
    k_range = range(k_min, k_max + 1)
    pool_info = build_candidate_pool(table, pool, min_feature_coverage, min_unique_values)
    complete, candidate_cols = pool_info["complete"], pool_info["candidate_cols"]

    if len(complete) < 4 or len(candidate_cols) < target_k:
        return {"status": "insufficient_data", "pool": pool, "pool_info": pool_info}

    search = greedy_forward_select(complete, candidate_cols, target_k, k_range,
                                   max_subset_size, GMM_N_INIT_SEARCH, random_state, min_subset_size)

    if search["best_gated_subset"] is None:
        return {"status": "no_gated_subset", "pool": pool, "pool_info": pool_info, "search": search}

    subset = search["best_gated_subset"]
    # Recompute the winning subset's curves at higher GMM n_init for a more reliable final report.
    from sklearn.preprocessing import StandardScaler
    X_final = StandardScaler().fit_transform(complete[subset].values)
    final_result = score_subset(X_final, target_k, k_range, GMM_N_INIT_FINAL, random_state)

    null_result = null_permutation_test(
        complete, candidate_cols, search["best_gated_combined_score"], target_k, k_range,
        max_subset_size, GMM_N_INIT_SEARCH, random_state, n_null, jobs, min_subset_size)

    bootstrap_result = bootstrap_stability(
        complete, subset, target_k, k_range, GMM_N_INIT_SEARCH, random_state, n_bootstrap, jobs)

    bgm_result = bayesian_gmm_effective_k(complete, subset, k_max, random_state)

    return {
        "status": "ok", "pool": pool, "pool_info": pool_info, "search": search,
        "subset": subset, "final_result": final_result,
        "null_test": null_result, "bootstrap": bootstrap_result, "bayesian_gmm": bgm_result,
        "n_cells": len(complete),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-csv", default=DEFAULT_MASTER_CSV_PATH)
    parser.add_argument("--stage", default="both", choices=["full", "behavioral", "both"])
    parser.add_argument("--target-k", type=int, default=DEFAULT_TARGET_K)
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--max-subset-size", type=int, default=DEFAULT_MAX_SUBSET_SIZE)
    parser.add_argument("--min-subset-size", type=int, default=DEFAULT_MIN_SUBSET_SIZE,
                        help="Greedy search won't stop before reaching this many features, even if "
                             "adding more doesn't improve the score -- prevents a single low-"
                             "cardinality feature from trivially 'winning' via discretization.")
    parser.add_argument("--min-feature-coverage", type=float, default=DEFAULT_MIN_FEATURE_COVERAGE)
    parser.add_argument("--min-unique-values", type=int, default=DEFAULT_MIN_UNIQUE_VALUES,
                        help="Candidate features with fewer distinct values than this are dropped "
                             "before the search -- prevents a near-discrete feature (e.g. a small "
                             "integer count) from trivially 'winning' via binning rather than real "
                             "multivariate structure.")
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def plot_bic_silhouette_curves(result: dict, outdir: Path, command: str,
                               fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    """BIC-by-k as the PRIMARY validation panel (larger, on top) with
    silhouette-by-k as a secondary, supporting panel underneath -- per the
    user's explicit preference (2026-08-11) to validate the classifier
    primarily on BIC, with silhouette as secondary justification, not the
    other way around. Both panels mark target_k the same way so the two
    criteria's agreement/disagreement is visually direct.
    """
    if result["status"] != "ok":
        return
    outdir.mkdir(parents=True, exist_ok=True)
    curves = result["final_result"]["curves"]
    ks = sorted(curves.keys())
    bic_vals = [curves[k]["bic"] for k in ks]
    sil_vals = [curves[k]["silhouette"] for k in ks]
    target_k = result.get("target_k", DEFAULT_TARGET_K)
    best_bic_k = ks[int(np.argmin(bic_vals))]
    best_sil_k = ks[int(np.argmax(sil_vals))]

    fig, (ax_bic, ax_sil) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [2, 1]})

    ax_bic.plot(ks, bic_vals, "o-", color="firebrick", lw=1.8, ms=6)
    ax_bic.axvline(target_k, color="black", ls="--", lw=1, alpha=0.6, label=f"target k={target_k}")
    if best_bic_k == target_k:
        ax_bic.scatter([target_k], [curves[target_k]["bic"]], s=140, facecolors="none",
                       edgecolors="mediumseagreen", linewidths=2, zorder=5, label="BIC minimum")
    ax_bic.set_ylabel("BIC (lower = better)")
    ax_bic.set_title(f"PRIMARY: GMM BIC by k -- {result['pool']} pool, "
                     f"{len(result['subset'])}-feature subset", fontsize=10)
    ax_bic.legend(loc="best", fontsize=7)

    ax_sil.plot(ks, sil_vals, "o-", color="steelblue", lw=1.4, ms=5)
    ax_sil.axvline(target_k, color="black", ls="--", lw=1, alpha=0.6)
    if best_sil_k == target_k:
        ax_sil.scatter([target_k], [curves[target_k]["silhouette"]], s=100, facecolors="none",
                       edgecolors="mediumseagreen", linewidths=2, zorder=5)
    ax_sil.set_xlabel("k")
    ax_sil.set_ylabel("silhouette")
    ax_sil.set_title("secondary / supporting: silhouette by k", fontsize=9)

    agree = "agree" if (best_bic_k == target_k and best_sil_k == target_k) else \
           ("BIC supports k, silhouette doesn't" if best_bic_k == target_k else
            "silhouette supports k, BIC doesn't" if best_sil_k == target_k else
            "neither criterion's own optimum is k")
    fig.suptitle(f"subset: {result['subset']}\nBIC optimum: k={best_bic_k}  |  "
                f"silhouette optimum: k={best_sil_k}  |  {agree}", fontsize=8, y=0.99)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    fig.text(0.5, 0.01, command, ha="center", va="bottom", fontsize=6, color="gray")
    fig.savefig(outdir / f"bic_silhouette_{result['pool']}.{fig_format}")
    plt.close(fig)


def print_stage_report(result: dict) -> None:
    pool = result["pool"]
    print(f"\n{'=' * 70}\nStage: {pool}\n{'=' * 70}")

    if result["status"] == "insufficient_data":
        print("Skipped: insufficient cells or candidate features.")
        return

    info = result["pool_info"]
    print(f"Candidate pool: {len(info['candidate_cols'])} features, {len(info['complete'])} cells "
         f"(dropped for zero variance: {info['zero_var_dropped']}; "
         f"dropped for low coverage: {info['coverage_dropped']}; "
         f"dropped for low cardinality: {info['low_cardinality_dropped']}; "
         f"excluded cells: {info['excluded_cell_ids']})")

    if result["status"] == "no_gated_subset":
        print(f"\n{NOT_SUPPORTED_MSG}")
        print("Greedy search path (feature added -> combined score, gate pass/fail):")
        for step in result["search"]["history"]:
            print(f"  +{step['added']}: sil_margin={step['silhouette_margin']:+.3f} "
                 f"bic_margin={step['bic_margin']:+.1f} gate={'PASS' if step['gate_passed'] else 'fail'}")
        return

    subset = result["subset"]
    final = result["final_result"]
    null_test = result["null_test"]
    bootstrap = result["bootstrap"]
    bgm = result["bayesian_gmm"]

    print(f"\nWinning subset ({len(subset)} features, n={result['n_cells']} cells): {subset}")
    print(f"\nSilhouette by k: " + ", ".join(f"{k}={v['silhouette']:.3f}" for k, v in sorted(final["curves"].items())))
    print(f"BIC by k:        " + ", ".join(f"{k}={v['bic']:.1f}" for k, v in sorted(final["curves"].items())))
    print(f"silhouette_margin (target_k vs best of rest): {final['silhouette_margin']:+.3f}")
    print(f"bic_margin (target_k vs best of rest):        {final['bic_margin']:+.1f}")

    print(f"\nNull-permutation test: real best-gated score's percentile among {null_test['n_null']} "
         f"random-column-order searches = "
         f"{null_test['real_score_percentile']:.1f}" if null_test["real_score_percentile"] is not None
         else "\nNull-permutation test: could not compute (no finite null scores)")
    print(f"Bootstrap stability: k=3 wins in {bootstrap['fraction_k_wins']:.1%} of "
         f"{bootstrap['n_bootstrap']} resamples")
    print(f"Bayesian GMM (Dirichlet-process) effective k: {bgm['effective_k']} "
         f"(weights: {[f'{w:.3f}' for w in bgm['weights']]}, threshold={bgm['weight_threshold']:.4f})")

    strong_null = null_test["real_score_percentile"] is not None and null_test["real_score_percentile"] >= 95.0
    strong_bootstrap = bootstrap["fraction_k_wins"] >= 0.80
    agrees_bgm = bgm["effective_k"] == result.get("target_k", DEFAULT_TARGET_K)
    print(f"\nVerdict: null-test {'PASSES' if strong_null else 'does NOT clearly pass'} (>=95th pct), "
         f"bootstrap {'PASSES' if strong_bootstrap else 'does NOT clearly pass'} (>=80%), "
         f"Bayesian GMM {'agrees' if agrees_bgm else 'does NOT agree'} on k.")
    if not (strong_null and strong_bootstrap):
        print(f"-> {NOT_SUPPORTED_MSG}")


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.master_csv, index_col="cell_id")

    pools = ["full", "behavioral"] if args.stage == "both" else [args.stage]
    results = {}
    for pool in pools:
        result = run_stage(
            table, pool, args.target_k, args.k_min, args.k_max, args.max_subset_size,
            args.min_feature_coverage, args.n_null, args.n_bootstrap, args.jobs, args.random_state,
            args.min_subset_size, args.min_unique_values)
        result["target_k"] = args.target_k
        results[pool] = result
        print_stage_report(result)
        if not args.no_plot and result["status"] == "ok":
            plot_bic_silhouette_curves(result, Path(args.figures_dir), "python " + " ".join(sys.argv),
                                       args.figure_format)
            print(f"Figure written to {args.figures_dir}/bic_silhouette_{pool}.{args.figure_format}")

    if "full" in results and "behavioral" in results:
        print(f"\n{'=' * 70}\nStage comparison\n{'=' * 70}")
        for pool in ("full", "behavioral"):
            r = results[pool]
            if r["status"] == "ok":
                print(f"  {pool}: subset size={len(r['subset'])}, "
                     f"silhouette_margin={r['final_result']['silhouette_margin']:+.3f}, "
                     f"null_pct={r['null_test']['real_score_percentile']}, "
                     f"bootstrap={r['bootstrap']['fraction_k_wins']:.1%}")
            else:
                print(f"  {pool}: {r['status']}")


if __name__ == "__main__":
    main()
