"""Current injection protocol: wait (0nA) / depolarizing pulse / post (0nA),
run on one, several, or all cells in --params-dir. Plots the voltage trace
and the injected current stacked in one figure per cell.

Each selected cell is simulated exactly ONCE at the given --dt. If a cell
blows up, this script reports it and moves on to the next cell -- it does
NOT retry automatically at a smaller dt. Rerun just that cell once you've
picked a smaller dt:

    python inject_current_pulse_v1.py --cells CELLID --dt 0.01

By default, each selected cell's initial condition is loaded from the
steady-state cache (see generate_steady_state.py) keyed by its own cell ID,
so the injection starts from the cell's natural firing state instead of the
biased startup transient. A cell with no cached steady state is skipped
with a warning -- run generate_steady_state.py for it first, or pass
--no-cache to start every cell from the default startup condition instead.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import load_all_cells, get_cached_y_ss

DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "figures" / "injection"
DEFAULT_DT_MS = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Current injection protocol: wait (0nA) / depolarizing pulse / '
                     'post (0nA), run on one, several, or all cells in --params-dir. '
                     'Plots the voltage trace and the injected current stacked in one '
                     'figure per cell.')
    p.add_argument('--params-dir', default=DEFAULT_PARAMS_DIR,
                    help='Directory containing 40-value cell parameter files.')
    p.add_argument('--cells', nargs='+', default=None,
                    help='Specific cell ID(s) (parameter filename stem, e.g. 0EFZ6A) '
                         'to run. Default: every cell found in --params-dir.')
    p.add_argument('--no-cache', action='store_true',
                    help="Don't use the steady-state cache -- start every cell from "
                         "the default startup condition instead of skipping cells "
                         "with no cached steady state.")
    p.add_argument('--amplitude', type=float, default=1.0, help='Pulse amplitude in nA (depolarizing = positive)')
    p.add_argument('--wait-s', type=float, default=0.2, help='default 200ms')
    p.add_argument('--pulse-s', type=float, default=0.2, help='default 200ms')
    p.add_argument('--post-s', type=float, default=0.2, help='default 200ms')
    p.add_argument('--iapp-mode', choices=['add', 'replace'], default='add',
                    help="'add': pulse is added on top of the file's own Iapp (p[39]). "
                         "'replace': ignore the file's Iapp entirely, use exactly "
                         "0/amplitude/0 nA. Default: add.")
    p.add_argument('--temp', type=float, default=25.0)
    p.add_argument('--reftemp', type=float, default=10.0)
    p.add_argument('--dt', type=float, default=DEFAULT_DT_MS,
                    help='Integration timestep in ms. Each cell is simulated once at '
                         'this dt -- if a cell blows up, rerun just that cell (via '
                         '--cells) with a smaller --dt.')
    p.add_argument('--outdir', default=DEFAULT_OUTDIR,
                    help='Directory to write one {cell_id}_injection.png per cell.')
    return p.parse_args()


def run_injection(cell_id: str, p: np.ndarray, cis, args: argparse.Namespace,
                  outdir: Path, command: str) -> Path:
    base_iapp = p[39]
    t_wait_end = args.wait_s * 1000.0
    t_pulse_end = t_wait_end + args.pulse_s * 1000.0
    nsecs_total = args.wait_s + args.pulse_s + args.post_s

    def pulse_current(t_ms):
        if t_wait_end <= t_ms < t_pulse_end:
            return args.amplitude
        return 0.0

    if args.iapp_mode == 'add':
        Iapp_func = lambda t_ms: base_iapp + pulse_current(t_ms)
    else:  # replace
        Iapp_func = lambda t_ms: pulse_current(t_ms)

    try:
        t_ms, states = simulate(p, nsecs_total, args.temp, dt=args.dt,
                                 reftemp=args.reftemp, cis=cis, Iapp_func=Iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise RuntimeError(f"dt={args.dt}: {exc}") from exc

    if not np.all(np.isfinite(states)):
        raise RuntimeError(f"dt={args.dt}: non-finite trajectory")

    V = states[:, 0]
    I_injected = np.array([Iapp_func(t) for t in t_ms])

    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
    ax_v.plot(t_ms, V, color='black', lw=0.8)
    ax_v.set_ylabel('V (mV)')
    ax_v.set_title(f'{cell_id} -- current injection ({args.iapp_mode})')

    ax_i.plot(t_ms, I_injected, color='red', lw=1.2)
    ax_i.set_ylabel('Iapp (nA)')
    ax_i.set_xlabel('time (ms)')
    pulse_end_level = base_iapp + args.amplitude if args.iapp_mode == 'add' else args.amplitude
    ax_i.set_ylim(min(0, base_iapp, pulse_end_level) - 0.2,
                  max(0, base_iapp, pulse_end_level) + 0.2)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(0.5, 0.01, command, ha='center', va='bottom',
             fontsize=6, family='monospace', color='dimgray', wrap=True)
    outpath = outdir / f'{cell_id}_injection.png'
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_cells = load_all_cells(params_dir)
    if args.cells:
        unknown = sorted(set(args.cells) - set(all_cells))
        if unknown:
            raise SystemExit(f"Unknown cell id(s): {unknown}\n"
                              f"Available cells in {params_dir}: {sorted(all_cells)}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Running current injection on {len(cells)} of {len(all_cells)} cell(s) "
          f"from {params_dir} at dt={args.dt} ms")

    written, skipped, failed = [], [], []
    for cell_id, p in cells.items():
        cis = None
        if not args.no_cache:
            cis = get_cached_y_ss(cell_id, p)
            if cis is None:
                print(f"  {cell_id}: skipped -- no cached steady state. Run "
                      f"generate_steady_state.py --cells {cell_id} first, or pass "
                      f"--no-cache to start from the default startup condition.")
                skipped.append(cell_id)
                continue

        try:
            outpath = run_injection(cell_id, p, cis, args, outdir, command)
        except RuntimeError as exc:
            print(f"  {cell_id}: blew up at dt={args.dt} ({exc}). Rerun with "
                  f"--cells {cell_id} --dt <smaller>.")
            failed.append(cell_id)
            continue

        print(f"  {cell_id}: saved {outpath}")
        written.append(cell_id)

    print(f"\n{len(written)} figure(s) written to {outdir}/, "
          f"{len(skipped)} skipped (no cache), {len(failed)} failed (blew up).")
    if skipped:
        print(f"Skipped: {skipped}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == '__main__':
    main()
