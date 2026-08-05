"""Helpers for loading and retrieving cached steady-state cell initial conditions."""

import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np

PARAMS_DIR = Path(__file__).resolve().parent / "models"
CACHE_PATH = Path(__file__).resolve().parent.parent / "cell_steady_states.pkl"


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
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict, cache_path: Path = CACHE_PATH) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


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
