import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from singlecell_model_v1 import simulate

def _parse_args():
    p = argparse.ArgumentParser(
        description='Current injection protocol: 1s wait (0nA) / 1s depolarizing pulse '
                     '/ 1s no current (0nA). Plots the voltage trace and the injected '
                     'current stacked in one figure.')
    p.add_argument('--params', required=True, help='Path to a single-cell parameter .txt file')
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
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--outfile', default='injection.png')
    return p.parse_args()

_args = _parse_args()

p = np.genfromtxt(_args.params)
base_iapp = p[39]

t_wait_end = _args.wait_s * 1000.0
t_pulse_end = t_wait_end + _args.pulse_s * 1000.0
nsecs_total = _args.wait_s + _args.pulse_s + _args.post_s

def pulse_current(t_ms):
    if t_wait_end <= t_ms < t_pulse_end:
        return _args.amplitude
    return 0.0

if _args.iapp_mode == 'add':
    Iapp_func = lambda t_ms: base_iapp + pulse_current(t_ms)
else:  # replace
    Iapp_func = lambda t_ms: pulse_current(t_ms)

print(f'Running injection protocol: mode={_args.iapp_mode}, base Iapp (file)={base_iapp} nA, '
      f'pulse amplitude={_args.amplitude} nA, total={nsecs_total}s '
      f'(wait {_args.wait_s}s / pulse {_args.pulse_s}s / post {_args.post_s}s)')

t_ms, states = simulate(p, nsecs_total, _args.temp, dt=_args.dt, reftemp=_args.reftemp, Iapp_func=Iapp_func)
V = states[:, 0]
I_injected = np.array([Iapp_func(t) for t in t_ms])

fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
ax_v.plot(t_ms, V, color='black', lw=0.8)
ax_v.set_ylabel('V (mV)')
ax_v.set_title(f'{os.path.basename(_args.params)} -- current injection ({_args.iapp_mode})')

ax_i.plot(t_ms, I_injected, color='red', lw=1.2)
ax_i.set_ylabel('Iapp (nA)')
ax_i.set_xlabel('time (ms)')
ax_i.set_ylim(min(0, base_iapp, base_iapp + _args.amplitude if _args.iapp_mode == 'add' else _args.amplitude) - 0.2,
              max(0, base_iapp, base_iapp + _args.amplitude if _args.iapp_mode == 'add' else _args.amplitude) + 0.2)

fig.tight_layout()
fig.savefig(_args.outfile, dpi=200)
print(f'saved: {_args.outfile}')
