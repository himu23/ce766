"""
Numba versions of GR4J and the reservoir simulation in models.py (same equations),
used for the optimisation and the uncertainty ensemble.

Policy vector x (27 levers)
  x[0:12]   monthly storage trigger  k_m  (fraction of capacity)
  x[12:24]  monthly rationing factor a_m  (supply as a fraction of demand when S < k_m * Smax)
  x[24]     b_dry  fraction of the Tennant e-flow released in Oct-May
  x[25]     b_mon  fraction of the Tennant e-flow released in Jun-Sep
  x[26]     g      extra factor on the e-flow release when S < k_m * Smax
"""
import numpy as np
from numba import njit, prange

N_LEV = 27
XL = np.r_[np.zeros(12), np.full(12, 0.5), 0.0, 0.0, 0.0]   # cuts deeper than 50 % not allowed
XU = np.ones(N_LEV)


@njit(cache=True, fastmath=False)
def gr4j_nb(P, E, x1, x2, x3, x4, s0=0.5, r0=0.5):
    n1, n2 = int(np.ceil(x4)), int(np.ceil(2 * x4))

    def sh1(t):
        if t <= 0:
            return 0.0
        return (t / x4) ** 2.5 if t < x4 else 1.0

    def sh2(t):
        if t <= 0:
            return 0.0
        if t <= x4:
            return 0.5 * (t / x4) ** 2.5
        if t < 2 * x4:
            return 1 - 0.5 * (2 - t / x4) ** 2.5
        return 1.0

    uh1 = np.array([sh1(i) - sh1(i - 1) for i in range(1, n1 + 1)])
    uh2 = np.array([sh2(i) - sh2(i - 1) for i in range(1, n2 + 1)])
    b1, b2 = np.zeros(n1), np.zeros(n2)
    S, R = s0 * x1, r0 * x3
    Q = np.empty(len(P))
    for t in range(len(P)):
        p, e = P[t], E[t]
        sr = S / x1
        if p >= e:
            pn = p - e
            tw = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1 - sr * sr) * tw / (1 + sr * tw)
            es = 0.0
        else:
            pn = 0.0
            tw = np.tanh(min((e - p) / x1, 13.0))
            es = S * (2 - sr) * tw / (1 + (1 - sr) * tw)
            ps = 0.0
        S = S - es + ps
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        for k in range(n1 - 1):
            b1[k] = b1[k + 1] + 0.9 * pr * uh1[k]
        b1[n1 - 1] = 0.9 * pr * uh1[n1 - 1]
        for k in range(n2 - 1):
            b2[k] = b2[k + 1] + 0.1 * pr * uh2[k]
        b2[n2 - 1] = 0.1 * pr * uh2[n2 - 1]
        F = x2 * (R / x3) ** 3.5
        R = max(0.0, R + b1[0] + F)
        qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
        R -= qr
        Q[t] = qr + max(0.0, b2[0] + F)
    return Q


@njit(cache=True)
def simulate(x, I, P, E, month, smax, demand, T, a_full):
    """Daily balance; T = daily e-flow target (ML/d). Returns storage, supply, release, spill, hedge flag."""
    n = len(I)
    S = np.empty(n); D = np.empty(n); R = np.empty(n); Sp = np.empty(n); H = np.zeros(n)
    s = smax
    for t in range(n):
        m = month[t] - 1
        mon = 5 <= m <= 8
        S[t] = s
        area = a_full * np.cbrt(max(s, 0.0) / smax) ** 2
        av = max(s + I[t] + (P[t] - E[t]) * area, 0.0)
        dq = demand
        rq = T[t] * (x[25] if mon else x[24])
        if s < x[m] * smax:
            dq *= x[12 + m]
            rq *= x[26]
            H[t] = 1.0
        d = min(dq, av)
        r = min(rq, av - d)
        e = av - d - r
        sp = max(0.0, e - smax)
        s = min(e, smax)
        D[t], R[t], Sp[t] = d, r, sp
    return S, D, R, Sp, H


@njit(cache=True)
def measures(D, R, Sp, month, demand, T):
    """0 reliability, 1 worst 7-day mean supply deficit (frac of demand),
       2 e-flow adequacy of the worse-served season = min(10, 11), 3 resilience,
       4 mean event deficit (ML), 5 number of events, 6 volumetric shortfall (frac),
       7 e-flow compliance (days), 8 max daily deficit frac, 9 mean annual spill (ML/yr),
       10 dry-season (Oct-May) and 11 monsoon (Jun-Sep) mean daily e-flow adequacy"""
    n = len(D)
    nf = 0; rec = 0; nev = 0; tot = 0.0; worst = 0.0; roll = 0.0
    comp = 0; mx = 0.0; spill = 0.0
    ad = np.zeros(2); nd = np.zeros(2)
    prev = False
    for t in range(n):
        df = max(demand - D[t], 0.0)
        fail = df > 1e-6
        tot += df
        roll += df
        if t >= 7:
            roll -= max(demand - D[t - 7], 0.0)
        worst = max(worst, roll)
        mx = max(mx, df)
        if fail:
            nf += 1
            if not prev:
                nev += 1
        elif prev:
            rec += 1
        prev = fail
        j = 1 if 6 <= month[t] <= 9 else 0
        q = R[t] + Sp[t]
        ad[j] += min(q / T[t], 1.0) if T[t] > 0 else 1.0
        nd[j] += 1
        if q >= T[t] - 1e-6:
            comp += 1
        spill += Sp[t]
    ad /= nd
    return np.array([1 - nf / n, worst / (7 * demand), ad.min(), rec / nf if nf else 1.0,
                     tot / nev if nev else 0.0, nev, tot / (n * demand), comp / n,
                     mx / demand, spill / n * 365.25, ad[0], ad[1]])


@njit(cache=True)
def evaluate(x, I, P, E, month, smax, demand, T, a_full):
    S, D, R, Sp, H = simulate(x, I, P, E, month, smax, demand, T, a_full)
    return measures(D, R, Sp, month, demand, T)


@njit(parallel=True, cache=True)
def evaluate_pop(X, I, P, E, month, smax, demand, T, a_full):
    out = np.empty((X.shape[0], 12))
    for k in prange(X.shape[0]):
        out[k] = evaluate(X[k], I, P, E, month, smax, demand, T, a_full)
    return out
