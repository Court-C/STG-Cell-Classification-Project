"""Helpers for loading and retrieving cached steady-state cell initial conditions."""

import gzip
import pathlib
import pickle
import platform
import re
from pathlib import Path
from typing import Optional

import numpy as np

# Some committed caches (e.g. cell_held_injected_grid.pkl) were pickled on
# Windows and embed WindowsPath instances (e.g. inside run_args), which
# pickle.load refuses to reconstruct on macOS/Linux ("cannot instantiate
# WindowsPath on your system"). Every cache-reading script imports this
# module before unpickling anything, so aliasing WindowsPath to PosixPath
# here -- the standard cross-platform-unpickling workaround -- covers all
# of them in one place rather than patching each pickle.load call.
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

PARAMS_DIR = Path(__file__).resolve().parent / "models"
CACHE_PATH = Path(__file__).resolve().parent.parent / "cell_steady_states.pkl"

# Spacing (ms) that generate_steady_state.py's _downsample_trace stores
# burnin_v at -- must match that constant. burnin_t is never itself written
# to disk (see _strip_for_storage): for a 50000-point trace it was pure
# redundancy, an evenly-spaced arange fully determined by this spacing and
# len(burnin_v), so it's reconstructed on load instead (_reconstruct_burnin_t).
MAX_STORED_TRACE_DT_MS = 0.1


def _reconstruct_burnin_t(entry: dict) -> dict:
    v = entry.get("burnin_v")
    if v is None or entry.get("burnin_t") is not None:
        return entry
    return {**entry, "burnin_t": np.arange(len(v), dtype=np.float64) * MAX_STORED_TRACE_DT_MS}


def _strip_for_storage(entry: dict) -> dict:
    """Drops burnin_t (see _reconstruct_burnin_t) and stores burnin_v as
    float32 -- voltage traces don't need 64-bit precision for what they're
    used for (peak-finding, plotting, coarse ISI stats). Combined with gzip
    in save_cache, this cuts cell_steady_states.pkl by ~80% (55MB -> ~12MB
    measured directly on the real cache) without losing any information
    that's actually used downstream.
    """
    v = entry.get("burnin_v")
    if v is None:
        return entry
    e = dict(entry)
    e.pop("burnin_t", None)
    e["burnin_v"] = np.asarray(v, dtype=np.float32)
    return e


def load_cell_params(filepath: Path) -> np.ndarray:
    text = filepath.read_text()
    values = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)]
    arr = np.array(values, dtype=np.float64)
    if arr.size != 40:
        raise ValueError(f"{filepath}: expected 40 params, got {arr.size}")
    return arr


def load_all_cells(params_dir: Path = PARAMS_DIR) -> dict[str, np.ndarray]:
    cells = {}
    for f in sorted(params_dir.glob("*.txt")):
        cells[f.stem] = load_cell_params(f)
    return cells


def load_cache(cache_path: Path = CACHE_PATH) -> dict:
    if not cache_path.exists():
        return {}
    with gzip.open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return {cell_id: _reconstruct_burnin_t(entry) for cell_id, entry in cache.items()}


def save_cache(cache: dict, cache_path: Path = CACHE_PATH) -> None:
    stored = {cell_id: _strip_for_storage(entry) for cell_id, entry in cache.items()}
    with gzip.open(cache_path, "wb", compresslevel=6) as f:
        pickle.dump(stored, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_cached_state(cell_id: str, params: Optional[np.ndarray] = None,
                     cache_path: Path = CACHE_PATH) -> Optional[dict]:
    cache = load_cache(cache_path)
    entry = cache.get(cell_id)
    if entry is None:
        return None
    if entry.get("status") != "ok":
        # e.g. "drift_warning" entries still carry a y_ss even though the
        # limit cycle wasn't fully verified -- only "ok" is trustworthy.
        return None
    if params is not None and not np.array_equal(entry.get("params"), params):
        return None
    return entry


def get_cached_y_ss(cell_id: str, params: Optional[np.ndarray] = None,
                    cache_path: Path = CACHE_PATH) -> Optional[np.ndarray]:
    entry = get_cached_state(cell_id, params, cache_path)
    if entry is None:
        return None
    return entry.get("y_ss")
