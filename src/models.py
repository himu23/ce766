"""
Core models: GR4J rainfall-runoff, daily reservoir simulation with a
parameterised hedging rule, and performance measures.
"""
import math
import numpy as np
import pandas as pd

# ============================================================== GR4J
# Perrin, Michel & Andreassian (2003), J. Hydrol. 279, 275-289.
GR4J_BOUNDS = [(10, 2500),    # X1 production store capacity (mm)
               (-10, 5),      # X2 groundwater exchange coefficient (mm/day)
               (1, 500),      # X3 routing store capacity (mm)
               (0.5, 6)]      # X4 unit hydrograph time base (days)


def _sh1(t, x4):
    if t <= 0:
        return 0.0
    return (t / x4) ** 2.5 if t < x4 else 1.0


def _sh2(t, x4):
    if t <= 0:
        return 0.0
    if t <= x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1 - 0.5 * (2 - t / x4) ** 2.5
    return 1.0


def _uh_ordinates(x4):
    n1, n2 = int(math.ceil(x4)), int(math.ceil(2 * x4))
    uh1 = [_sh1(i, x4) - _sh1(i - 1, x4) for i in range(1, n1 + 1)]
    uh2 = [_sh2(i, x4) - _sh2(i - 1, x4) for i in range(1, n2 + 1)]
    return uh1, uh2


def _gr4j_py(P, E, x1, x2, x3, x4, s0, r0):
    uh1, uh2 = _uh_ordinates(x4)
    n1, n2 = len(uh1), len(uh2)
    b1, b2 = [0.0] * n1, [0.0] * n2
    S, R = s0 * x1, r0 * x3
    tanh = math.tanh
    Q = [0.0] * len(P)
    for t in range(len(P)):
        p, e = P[t], E[t]
        if p >= e:
            pn = p - e
            tw = tanh(min(pn / x1, 13.0))
            sr = S / x1
            ps = x1 * (1 - sr * sr) * tw / (1 + sr * tw)
            es = 0.0
        else:
            pn = 0.0
            tw = tanh(min((e - p) / x1, 13.0))
            sr = S / x1
            es = S * (2 - sr) * tw / (1 + (1 - sr) * tw)
            ps = 0.0
        S = S - es + ps
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        # unit hydrographs as shifting buffers
        for k in range(n1 - 1):
            b1[k] = b1[k + 1] + 0.9 * pr * uh1[k]
        b1[n1 - 1] = 0.9 * pr * uh1[n1 - 1]
        for k in range(n2 - 1):
            b2[k] = b2[k + 1] + 0.1 * pr * uh2[k]
        b2[n2 - 1] = 0.1 * pr * uh2[n2 - 1]
        q9, q1 = b1[0], b2[0]
        F = x2 * (R / x3) ** 3.5
        R = max(0.0, R + q9 + F)
        qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
        R -= qr
        Q[t] = qr + max(0.0, q1 + F)
    return Q


def gr4j(P, E, params, s0=0.5, r0=0.5):
    """Run GR4J. P, E in mm/day. Returns simulated runoff, mm/day (numpy array)."""
    x1, x2, x3, x4 = map(float, params)
    return np.asarray(_gr4j_py(list(map(float, P)), list(map(float, E)), x1, x2, x3, x4, s0, r0))


# ============================================================== metrics
def kge(sim, obs):
    m = ~(np.isnan(sim) | np.isnan(obs))
    s, o = sim[m], obs[m]
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), r, alpha, beta


def nse(sim, obs):
    m = ~(np.isnan(sim) | np.isnan(obs))
    return 1 - np.sum((sim[m] - obs[m]) ** 2) / np.sum((obs[m] - obs[m].mean()) ** 2)


# ============================================================== reservoir
def eflow_target(inflow_ML, dates, frac_monsoon, frac_dry, monsoon_months):
    """Tennant-type environmental flow, ML/day, from the mean annual flow."""
    maf = np.nanmean(inflow_ML)                      # mean daily flow, ML/day
    mon = np.isin(dates.month, monsoon_months)
    return np.where(mon, frac_monsoon, frac_dry) * maf


def simulate_reservoir(I, P_mm, E_mm, dates, cfg, policy=None, s0_frac=1.0):
    """
    Daily mass balance:
        S(t+1) = S(t) + I(t) + [P(t) - E(t)] * A(S(t)) - D(t) - R(t) - Sp(t)
    policy: None -> Standard Operating Policy (meet full demand while water lasts)
            dict(trigger=12 monthly storage fractions, alpha=12 monthly rationing factors,
                 eflow_mult=scalar) -> hedging rule
    Returns DataFrame with storage, supply, release, spill, deficits.
    """
    n = len(I)
    Smax = cfg.S_MAX_ML
    e_req = eflow_target(I, dates, cfg.EFLOW_FRAC_MONSOON, cfg.EFLOW_FRAC_DRY, cfg.MONSOON_MONTHS)
    S = np.empty(n + 1); S[0] = s0_frac * Smax
    D = np.empty(n); Rl = np.empty(n); Sp = np.empty(n); Dreq = np.full(n, cfg.D_TARGET_MLD)
    months = dates.month.values
    for t in range(n):
        A = cfg.A_LAKE_FULL_KM2 * (max(S[t], 0) / Smax) ** cfg.AREA_EXPONENT
        avail = S[t] + I[t] + (P_mm[t] - cfg.PAN_COEFF * E_mm[t]) * A
        avail = max(avail, 0.0)
        target_D = Dreq[t]
        target_R = e_req[t]
        if policy is not None:
            m = months[t] - 1
            if S[t] < policy["trigger"][m] * Smax:       # hedge when storage is low
                target_D *= policy["alpha"][m]
            target_R *= policy.get("eflow_mult", 1.0)
        D[t] = min(target_D, avail)
        Rl[t] = min(target_R, avail - D[t])
        s_next = avail - D[t] - Rl[t]
        Sp[t] = max(0.0, s_next - Smax)
        S[t + 1] = min(s_next, Smax)
    return pd.DataFrame({"storage_ML": S[:-1], "supply_ML": D, "supply_req_ML": Dreq,
                         "release_ML": Rl, "eflow_req_ML": e_req, "spill_ML": Sp},
                        index=dates)


def rrv(supply, demand, tol=1e-6):
    """Hashimoto, Stedinger & Loucks (1982) reliability, resilience, vulnerability."""
    fail = supply < demand - tol
    reliability = 1 - fail.mean()
    recov = np.sum(fail[:-1] & ~fail[1:])
    resilience = recov / fail.sum() if fail.sum() else 1.0
    # vulnerability: mean of the total deficit per failure event (as % of daily demand)
    deficit = np.where(fail, demand - supply, 0)
    events, cur = [], 0.0
    for f, d in zip(fail, deficit):
        if f:
            cur += d
        elif cur > 0:
            events.append(cur); cur = 0.0
    if cur > 0:
        events.append(cur)
    vul = np.mean(events) if events else 0.0
    max_def_frac = np.max(deficit / demand) if fail.any() else 0.0
    return dict(reliability=reliability, resilience=resilience,
                vulnerability_ML=vul, n_events=len(events), max_daily_deficit_frac=max_def_frac)


def performance(sim):
    out = rrv(sim.supply_ML.values, sim.supply_req_ML.values)
    downstream = sim.release_ML.values + sim.spill_ML.values     # spill also reaches the river
    out["eflow_compliance"] = np.mean(downstream >= sim.eflow_req_ML.values - 1e-6)
    ann_spill = sim.spill_ML.resample("YE").sum() / 1000           # MCM/yr
    out["mean_annual_spill_MCM"] = ann_spill.mean()
    out["max_daily_spill_ML"] = sim.spill_ML.max()
    out["supply_shortfall_pct"] = 100 * (1 - sim.supply_ML.sum() / sim.supply_req_ML.sum())
    return out
