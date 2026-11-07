"""Statistics for the DT+NDT and DT-only tabs. All dispersion is taken in log space (lognormal f_c,is)."""
import numpy as np
from scipy.stats import t as student_t

ALPHA_BETA = 0.8 * 3.8      # alpha_R * beta_tgt
V_MIN_DT = 0.08             # 8 % minimum CoV for cores alone (EN 1992-1-1 Annex A)
S_MIN_MPA = 2.0             # minimum model standard deviation (EN 13791), MPa


def k_n(n):
    """5 % fractile coefficient, V unknown (Table 1; EN 1990 Table D.1)."""
    return float(student_t.ppf(0.95, n - 1) * np.sqrt(1.0 + 1.0 / n))


def k_dn(n):
    """ULS design coefficient, V unknown (Table 1; EN 1990 Table D.2)."""
    return float(student_t.ppf(0.999, n - 1) * np.sqrt(1.0 + 1.0 / n))


def cov_from_zeta(z):
    return float(np.sqrt(np.expm1(z ** 2)))


def zeta_from_cov(V):
    return float(np.sqrt(np.log1p(V ** 2)))


def dt_only(fc):
    """DT-only analysis of normalised core strengths (Eqs. 5-7, 15, 16)."""
    fc = np.asarray(fc, dtype=float)
    n = fc.size
    ln = np.log(fc)
    lam = float(ln.mean())
    z_meas = float(np.std(ln, ddof=1))
    z_min = zeta_from_cov(V_MIN_DT)
    z = max(z_meas, z_min)
    V = cov_from_zeta(z)
    kn = k_n(n)
    kdn = k_dn(n)
    fcm = float(fc.mean())
    fck = fcm * np.exp(-kn * V)                       # Eq. (7)
    return {
        "n": n,
        "fcm": fcm,
        "s_fc": float(np.std(fc, ddof=1)),
        "lambda": lam,
        "zeta_measured": z_meas,
        "zeta": z,
        "V_measured": cov_from_zeta(z_meas),
        "V": V,
        "V_min_governs": bool(z_min > z_meas),
        "k_n": kn,
        "k_dn": kdn,
        "V_corr": V * kdn / ALPHA_BETA,              # Eq. (5)
        "fck": float(fck),
        "bias": float(fcm / fck),                     # Eq. (6)
    }


def calibrated(fc_n, ln_pred_n, ln_pred_m, p, shift=False):
    """Combined DT+NDT statistics with the 2 MPa floor and the DT-only lower bound"""
    fc_n = np.asarray(fc_n, dtype=float)
    ln_pred_n = np.asarray(ln_pred_n, dtype=float)
    ln_pred_m = np.asarray(ln_pred_m, dtype=float)
    n, m = fc_n.size, ln_pred_m.size

    resid = np.log(fc_n) - ln_pred_n
    delta = float(resid.mean()) if shift else 0.0
    ln_n = ln_pred_n + delta
    ln_m = ln_pred_m + delta
    resid = resid - delta

    lam = float(ln_m.mean())                                          # Eq. (9)
    z_test = float(np.std(ln_m, ddof=1))                              # Eq. (11)
    z_theta_raw = float(np.sqrt(np.sum(resid ** 2) / (n - p)))        # Eq. (12)
    z_floor = zeta_from_cov(S_MIN_MPA / fc_n.mean())
    z_theta = max(z_theta_raw, z_floor)
    z_cal = float(np.hypot(z_test, z_theta))                          # Eq. (10)

    den = z_theta ** 4 / (n - p) + (z_test ** 4 / (m - 1) if m > 1 else 0.0)
    n_eff = int(round((z_test ** 2 + z_theta ** 2) ** 2 / den))       # Eq. (13)
    n_eff = max(n_eff, 1)
    k_eff = k_n(n_eff + 1)          # Table 1 with n replaced by n_eff + 1

    dt = dt_only(fc_n)
    z = max(z_cal, dt["zeta"])
    V = cov_from_zeta(z)                                              # Eq. (14)
    kdn = k_dn(n)
    fck = float(np.exp(lam - k_eff * z))
    mean = float(np.exp(lam + z ** 2 / 2))

    return {
        "n": n, "m": m, "p": p,
        "shift": delta,
        "lambda": lam,
        "zeta_test": z_test,
        "zeta_theta_raw": z_theta_raw,
        "zeta_theta_floor": z_floor,
        "zeta_theta": z_theta,
        "floor_governs": bool(z_floor > z_theta_raw),
        "zeta_cal": z_cal,
        "V_cal": cov_from_zeta(z_cal),
        "V_dt": dt["V"],
        "dt_governs": bool(dt["zeta"] > z_cal),
        "zeta": z,
        "V": V,
        "n_eff": n_eff,
        "k_n_eff": k_eff,
        "k_dn": kdn,
        "V_corr": V * kdn / ALPHA_BETA,
        "fck": fck,
        "fcm": mean,
        "bias": mean / fck,
        "pred_n": np.exp(ln_n).tolist(),
        "pred_m": np.exp(ln_m).tolist(),
        "dt_only": dt,
    }
