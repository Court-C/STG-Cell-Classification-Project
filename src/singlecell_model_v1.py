"""
Standalone single-compartment conductance-based neuron model, with the IMI
(modulator-activated inward) current.

Transcribed directly from singleCompTemperature_imi_realclean.java
(Alonso/Marder lab). This file has NO dependency on the rest of that
project -- it only needs numpy -- so it can be shared as-is.

State vector (14 values), in this exact order:
    0: V      membrane potential (mV)
    1: NaM    Na activation
    2: NaH    Na inactivation
    3: CaTM   CaT activation
    4: CaTH   CaT inactivation
    5: CaSM   CaS activation
    6: CaSH   CaS inactivation
    7: HM     H activation
    8: KdM    Kd activation
    9: KCaM   KCa activation
    10: AM    A activation
    11: AH    A inactivation
    12: IMIM  IMI activation
    13: IntCa internal calcium concentration (uM)

Parameter vector (40 values), in this exact order:
    0: gNa, 1: gCaT, 2: gCaS, 3: gA, 4: gKCa, 5: gKd, 6: gH, 7: gL, 8: gImi
       (maximal conductances)
    9: EL, 10: ENa, 11: EKd (also used for EKCa and EA), 12: EH
       (reversal potentials; ECaT/ECaS come from the Ca Nernst equation instead)
    13-15: q10_gNa, q10_gNa_m, q10_gNa_h
    16-18: q10_gCaT, q10_gCaT_m, q10_gCaT_h
    19-21: q10_gCaS, q10_gCaS_m, q10_gCaS_h
    22-24: q10_gA, q10_gA_m, q10_gA_h
    25-27: q10_gKCa, q10_gKCa_m, q10_gKCa_h
    28-30: q10_gKdr, q10_gKdr_m, q10_gKdr_h
    31-33: q10_gH, q10_gH_m, q10_gH_h
    34: q10_g_leak
    35-36: q10_g_imi, q10_g_imi_m
    37: q10_tau_Ca
    38: tauIntCa (base calcium buffer time constant, ms)
    39: Iapp (applied/injected current, nA)

Fixed constants (not parameters, hardcoded exactly as in the Java source):
    C = 1.0 uF/cm^2            membrane capacitance
    EIMI = -10 mV               IMI reversal potential
    caF = 0.94, Ca0 = 0.05      calcium buffering constants
    tauImi (base) = 20 ms       IMI activation time constant, BEFORE q10 scaling
    reftemp = 10                single-cell reference temperature (NOTE: the
                                 network model overrides this to 25 via
                                 setRefTemp(), but the single-cell integrator
                                 (integrateTempCompCell_imi_realclean_rk4.java)
                                 never calls that override -- so a standalone
                                 single-cell run uses reftemp=10 by default,
                                 unless you explicitly pass a different value)

Currents returned (in this order): [Na, CaT, CaS, A, KCa, Kd, H, L, IMI]
"""
import numpy as np


# ---------------------------------------------------------------------------
# Elementary functions (exact transcriptions from the Java source)
# ---------------------------------------------------------------------------
def boltzSS(V, A, B):
    return 1. / (1. + np.exp((V + A) / B))


def tauX(V, CT, DT, AT, BT):
    return CT - DT / (1. + np.exp((V + AT) / BT))


def spectau(V, CT, DT, AT, BT, AT2, BT2):
    return CT + DT / (np.exp((V + AT) / BT) + np.exp((V + AT2) / BT2))


def iIonic(g, m, h, q, V, Erev):
    return g * (m ** q) * h * (V - Erev)


def CaNernst(CaIn, temp):
    R = 8.314e3
    T = 273.15 + temp
    z = 2.0
    Far = 96485.33
    CaOut = 3000.0
    return ((R * T) / (z * Far)) * np.log(CaOut / CaIn)


# ---------------------------------------------------------------------------
# Derivatives (right-hand side of the ODE system)
# ---------------------------------------------------------------------------
def derivatives(state, p, temp, reftemp=10.0, Iapp_override=None):
    """
    state: length-14 array, see module docstring for order.
    p: length-40 parameter array, see module docstring for order.
    temp: simulation temperature (deg C).
    reftemp: reference temperature the q10 factors are defined relative to.
        Single-cell default is 10 (see module docstring) -- pass 25 if you
        specifically want to match the network model's convention instead.
    Iapp_override: if not None, use this value (nA) instead of p[39] for the
        applied current this step. Used for time-varying current injection.

    Returns dstate/dt, same length-14 order as state.
    """
    V, NaM, NaH, CaTM, CaTH, CaSM, CaSH, HM, KdM, KCaM, AM, AH, IMIM, IntCa = state

    CaRev = CaNernst(IntCa, temp)
    C = 1.0

    caF = 0.94
    Ca0 = 0.05
    tauImi_base = 20.0

    gNa, gCaT, gCaS, gA, gKCa, gKd, gH, gL, gImi = p[0:9]
    EL, ENa, EKd, EH = p[9], p[10], p[11], p[12]
    EKCa = EKd
    EA = EKd
    EIMI = -10.0
    ECaT = CaRev
    ECaS = CaRev

    (q10_gNa, q10_gNa_m, q10_gNa_h,
     q10_gCaT, q10_gCaT_m, q10_gCaT_h,
     q10_gCaS, q10_gCaS_m, q10_gCaS_h,
     q10_gA, q10_gA_m, q10_gA_h,
     q10_gKCa, q10_gKCa_m, q10_gKCa_h,
     q10_gKdr, q10_gKdr_m, q10_gKdr_h,
     q10_gH, q10_gH_m, q10_gH_h,
     q10_g_leak,
     q10_g_imi, q10_g_imi_m,
     q10_tau_Ca) = p[13:38]

    tauIntCa_base = p[38]
    Iapp = p[39] if Iapp_override is None else Iapp_override

    # --- steady states ---
    NaMinf = boltzSS(V, 25.5, -5.29)
    NaHinf = boltzSS(V, 48.9, 5.18)
    CaTMinf = boltzSS(V, 27.1, -7.20)
    CaTHinf = boltzSS(V, 32.1, 5.50)
    CaSMinf = boltzSS(V, 33.0, -8.1)
    CaSHinf = boltzSS(V, 60.0, 6.20)
    HMinf = boltzSS(V, 70.0, 6.0)
    KdMinf = boltzSS(V, 12.3, -11.8)
    KCaMinf = (IntCa / (IntCa + 3.0)) * boltzSS(V, 28.3, -12.6)
    AMinf = boltzSS(V, 27.2, -8.70)
    AHinf = boltzSS(V, 56.9, 4.90)
    IMIMinf = boltzSS(V, 55.0, -5.0)

    # --- time constants (base, then q10-scaled) ---
    dtemp = temp - reftemp
    tauNaM = tauX(V, 1.32, 1.26, 120.0, -25.0) * (q10_gNa_m ** (-dtemp / 10.))
    tauNaH = (tauX(V, 0.0, -0.67, 62.9, -10.0) * tauX(V, 1.50, -1.00, 34.9, 3.60)) * (q10_gNa_h ** (-dtemp / 10.))
    tauCaTM = tauX(V, 21.7, 21.3, 68.1, -20.5) * (q10_gCaT_m ** (-dtemp / 10.))
    tauCaTH = tauX(V, 105.0, 89.8, 55.0, -16.9) * (q10_gCaT_h ** (-dtemp / 10.))
    tauCaSM = spectau(V, 1.40, 7.00, 27.0, 10.0, 70.0, -13.0) * (q10_gCaS_m ** (-dtemp / 10.))
    tauCaSH = spectau(V, 60.0, 150.0, 55.0, 9.00, 65.0, -16.0) * (q10_gCaS_h ** (-dtemp / 10.))
    tauHM = tauX(V, 272.0, -1499.0, 42.2, -8.73) * (q10_gH_m ** (-dtemp / 10.))
    tauKdM = tauX(V, 7.20, 6.40, 28.3, -19.2) * (q10_gKdr_m ** (-dtemp / 10.))
    tauKCaM = tauX(V, 90.3, 75.1, 46.0, -22.7) * (q10_gKCa_m ** (-dtemp / 10.))
    tauAM = tauX(V, 11.6, 10.4, 32.9, -15.2) * (q10_gA_m ** (-dtemp / 10.))
    tauAH = tauX(V, 38.6, 29.2, 38.9, -26.5) * (q10_gA_h ** (-dtemp / 10.))
    # NOTE: tauImi's q10 exponent sign is POSITIVE, unlike every other tau
    # above (all negative). This matches the Java source exactly
    # (singleCompTemperature_imi_realclean.java: "tauImi = tauImi*
    # Math.pow(q10_g_imi_m, (temp-reftemp)/10.)") -- preserved as-is, not
    # "corrected", since the point is to match the original model exactly.
    tauImi = tauImi_base * (q10_g_imi_m ** (dtemp / 10.))

    tauIntCa = tauIntCa_base * (q10_tau_Ca ** (-dtemp / 10.))

    # --- conductances, q10-scaled (all positive exponent) ---
    gNa = gNa * (q10_gNa ** (dtemp / 10.))
    gCaT = gCaT * (q10_gCaT ** (dtemp / 10.))
    gCaS = gCaS * (q10_gCaS ** (dtemp / 10.))
    gA = gA * (q10_gA ** (dtemp / 10.))
    gKCa = gKCa * (q10_gKCa ** (dtemp / 10.))
    gKd = gKd * (q10_gKdr ** (dtemp / 10.))
    gH = gH * (q10_gH ** (dtemp / 10.))
    gL = gL * (q10_g_leak ** (dtemp / 10.))
    gImi = gImi * (q10_g_imi ** (dtemp / 10.))

    # --- currents ---
    iNa = iIonic(gNa, NaM, NaH, 3, V, ENa)
    iCaT = iIonic(gCaT, CaTM, CaTH, 3, V, ECaT)
    iCaS = iIonic(gCaS, CaSM, CaSH, 3, V, ECaS)
    iH = iIonic(gH, HM, 1, 1, V, EH)
    iKd = iIonic(gKd, KdM, 1, 4, V, EKd)
    iKCa = iIonic(gKCa, KCaM, 1, 4, V, EKCa)
    iA = iIonic(gA, AM, AH, 3, V, EA)
    iL = iIonic(gL, 1, 1, 1, V, EL)
    iIMI = iIonic(gImi, IMIM, 1, 1, V, EIMI)

    dV = (-(iNa + iCaT + iCaS + iH + iKd + iKCa + iA + iL + iIMI) + Iapp) / C
    dNaM = (NaMinf - NaM) / tauNaM
    dNaH = (NaHinf - NaH) / tauNaH
    dCaTM = (CaTMinf - CaTM) / tauCaTM
    dCaTH = (CaTHinf - CaTH) / tauCaTH
    dCaSM = (CaSMinf - CaSM) / tauCaSM
    dCaSH = (CaSHinf - CaSH) / tauCaSH
    dHM = (HMinf - HM) / tauHM
    dKdM = (KdMinf - KdM) / tauKdM
    dKCaM = (KCaMinf - KCaM) / tauKCaM
    dAM = (AMinf - AM) / tauAM
    dAH = (AHinf - AH) / tauAH
    dIMIM = (IMIMinf - IMIM) / tauImi
    dIntCa = (-caF * (iCaT + iCaS) - IntCa + Ca0) / tauIntCa

    return np.array([dV, dNaM, dNaH, dCaTM, dCaTH, dCaSM, dCaSH, dHM,
                      dKdM, dKCaM, dAM, dAH, dIMIM, dIntCa])


def get_currents(state, p, temp, reftemp=10.0):
    """Return the 9 currents [Na,CaT,CaS,A,KCa,Kd,H,L,IMI] for a given
    state/parameter vector, without advancing the ODE. state can be a
    single (14,) vector or a (T,14) array of a trajectory."""
    state = np.asarray(state)
    single = (state.ndim == 1)
    if single:
        state = state[None, :]

    V = state[:, 0]
    NaM, NaH = state[:, 1], state[:, 2]
    CaTM, CaTH = state[:, 3], state[:, 4]
    CaSM, CaSH = state[:, 5], state[:, 6]
    HM = state[:, 7]
    KdM = state[:, 8]
    KCaM = state[:, 9]
    AM, AH = state[:, 10], state[:, 11]
    IMIM = state[:, 12]
    IntCa = state[:, 13]

    CaRev = CaNernst(IntCa, temp)
    gNa, gCaT, gCaS, gA, gKCa, gKd, gH, gL, gImi = p[0:9]
    EL, ENa, EKd, EH = p[9], p[10], p[11], p[12]
    EKCa, EA = EKd, EKd
    EIMI = -10.0
    ECaT, ECaS = CaRev, CaRev

    q10_gNa = p[13]; q10_gCaT = p[16]; q10_gCaS = p[19]; q10_gA = p[22]
    q10_gKCa = p[25]; q10_gKdr = p[28]; q10_gH = p[31]; q10_g_leak = p[34]
    q10_g_imi = p[35]
    dtemp = temp - reftemp

    gNa = gNa * (q10_gNa ** (dtemp / 10.))
    gCaT = gCaT * (q10_gCaT ** (dtemp / 10.))
    gCaS = gCaS * (q10_gCaS ** (dtemp / 10.))
    gA = gA * (q10_gA ** (dtemp / 10.))
    gKCa = gKCa * (q10_gKCa ** (dtemp / 10.))
    gKd = gKd * (q10_gKdr ** (dtemp / 10.))
    gH = gH * (q10_gH ** (dtemp / 10.))
    gL = gL * (q10_g_leak ** (dtemp / 10.))
    gImi = gImi * (q10_g_imi ** (dtemp / 10.))

    iNa = iIonic(gNa, NaM, NaH, 3, V, ENa)
    iCaT = iIonic(gCaT, CaTM, CaTH, 3, V, ECaT)
    iCaS = iIonic(gCaS, CaSM, CaSH, 3, V, ECaS)
    iH = iIonic(gH, HM, 1, 1, V, EH)
    iKd = iIonic(gKd, KdM, 1, 4, V, EKd)
    iKCa = iIonic(gKCa, KCaM, 1, 4, V, EKCa)
    iA = iIonic(gA, AM, AH, 3, V, EA)
    iL = iIonic(gL, np.ones_like(V), np.ones_like(V), 1, V, EL)
    iIMI = iIonic(gImi, IMIM, np.ones_like(V), 1, V, EIMI)

    currents = [iNa, iCaT, iCaS, iA, iKCa, iKd, iH, iL, iIMI]
    if single:
        currents = [c[0] for c in currents]
    return currents


CURRENT_NAMES = ['Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H', 'L', 'IMI']

# Initial conditions, exactly matching cis_singlecomprealclean_imi.txt:
# V=-51, all gating variables=0, IntCa=5.
DEFAULT_CIS = np.array([-51.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.])


# ---------------------------------------------------------------------------
# Classic RK4 integrator
# ---------------------------------------------------------------------------
def rk4_step(state, dt, p, temp, reftemp=10.0, Iapp_override=None):
    k1 = derivatives(state, p, temp, reftemp, Iapp_override)
    k2 = derivatives(state + 0.5 * dt * k1, p, temp, reftemp, Iapp_override)
    k3 = derivatives(state + 0.5 * dt * k2, p, temp, reftemp, Iapp_override)
    k4 = derivatives(state + dt * k3, p, temp, reftemp, Iapp_override)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(p, nsecs, temp, dt=0.1, reftemp=10.0, cis=None, Iapp_func=None):
    """
    Run the model for nsecs seconds. Returns (t_ms, states) where states has
    shape (nsteps, 14).

    Iapp_func: optional callable t_ms -> current (nA). If given, overrides
        p[39] at every step (time-varying injection). If None, uses the
        constant p[39] from the parameter file throughout.
    """
    p = np.asarray(p, dtype=float)
    state = np.array(cis, dtype=float) if cis is not None else DEFAULT_CIS.copy()

    nsteps = int(round((1000.0 * nsecs) / dt))
    states = np.empty((nsteps, 14))
    t_ms = np.arange(nsteps) * dt

    for i in range(nsteps):
        Iapp_override = Iapp_func(t_ms[i]) if Iapp_func is not None else None
        states[i] = state
        state = rk4_step(state, dt, p, temp, reftemp, Iapp_override)

    return t_ms, states


# ---------------------------------------------------------------------------
# CLI: run a plain control simulation and plot the voltage trace
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(
        description='Standalone single-cell conductance-based model (control run, '
                     'constant Iapp from the parameter file). Produces a voltage trace plot.')
    ap.add_argument('--params', required=True, help='Path to a 40-value single-cell parameter file')
    ap.add_argument('--nsecs', type=float, default=0.2, help='default 200ms')
    ap.add_argument('--temp', type=float, default=25.0)
    ap.add_argument('--reftemp', type=float, default=10.0,
                     help='Reference temperature for q10 scaling (single-cell default: 10)')
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--outfile', default='trace.png')
    args = ap.parse_args()

    p = np.genfromtxt(args.params)
    t_ms, states = simulate(p, args.nsecs, args.temp, dt=args.dt, reftemp=args.reftemp)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ms, states[:, 0], color='black', lw=0.8)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title(f'{args.params} -- control (Iapp={p[39]} nA)')
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=200)
    print(f'saved: {args.outfile}')
