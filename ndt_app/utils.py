import numpy as np
import pandas as pd


#  Normalise raw core strengths to the 150 x 300 mm reference cylinder

_F100 = {1.0: 0.96 / 1.10, 2.0: 0.92 / 0.95, 3.0: 1.05}
_R1 = _F100[1.0] / _F100[2.0]          # 1:1 / 2:1 ratio of the 100 mm class
_R3 = _F100[3.0] / _F100[2.0]          # 3:1 / 2:1 ratio of the 100 mm class

ANCHORS = {
    50:  dict(_F100),
    75:  {1.0: 0.98 / 1.06, 2.0: 1.00 / 1.06, 3.0: (1.00 / 1.06) * _R3},
    100: dict(_F100),
    150: {1.0: 1.00 * _R1, 2.0: 1.00, 3.0: 1.00 / 0.95},
}

# Nominal diameter classes (upper bounds, mm)
_CLASS_EDGES = [(62.5, 50), (87.5, 75), (125.0, 100), (np.inf, 150)]

# Range of geometries covered by the databases and Reineck
D_EVIDENCED = (44.0, 160.0)     # mm
LD_MIN_EXTRAP = 0.5             # below this H/D the factor is held constant
LD_MAX = 3.0                    # above this H/D the factor is held constant


def diameter_class(D):
    for upper, cls in _CLASS_EDGES:
        if D < upper:
            return cls
    return 150


def conversion_factor(D, H):
    """Return (factor, nominal class, H/D, note) for one core."""
    cls = diameter_class(D)
    a = ANCHORS[cls]
    ld = H / D
    notes = []

    if ld < 1.0:
        # Shorter cores test stronger: extend the 1:1 to 2:1 slope (conservative)
        ld_use = max(ld, LD_MIN_EXTRAP)
        slope = a[2.0] - a[1.0]
        f = a[1.0] - slope * (1.0 - ld_use)
        notes.append("H/D below 1, factor extrapolated" if ld >= LD_MIN_EXTRAP
                     else f"H/D below {LD_MIN_EXTRAP}, factor held at H/D = {LD_MIN_EXTRAP}")
    elif ld <= 2.0:
        f = a[1.0] + (a[2.0] - a[1.0]) * (ld - 1.0)
    elif ld <= LD_MAX:
        f = a[2.0] + (a[3.0] - a[2.0]) * (ld - 2.0)
    else:
        f = a[3.0]
        notes.append(f"H/D above {LD_MAX:.0f}, factor held at H/D = {LD_MAX:.0f}")

    if not (D_EVIDENCED[0] <= D <= D_EVIDENCED[1]):
        notes.append(f"diameter outside the {D_EVIDENCED[0]:.0f}–{D_EVIDENCED[1]:.0f} mm range of the conversion factors, "
                     f"factors for the nearest core size ({cls} mm) used")

    return float(f), cls, float(ld), "; ".join(notes)


def normalise_to_150x300(diam_mm, height_mm, fc_raw_mpa) -> pd.DataFrame:
    """Normalise core strengths to the 150 x 300 mm reference cylinder."""
    d = np.asarray(diam_mm, dtype=float)
    h = np.asarray(height_mm, dtype=float)
    fc0 = np.asarray(fc_raw_mpa, dtype=float)

    if not (d.size == h.size == fc0.size):
        raise ValueError("diameter, height and fc arrays must be the same length")
    if np.any(d <= 0) or np.any(h <= 0) or np.any(fc0 <= 0):
        raise ValueError("core diameter, height and strength must all be positive")

    rows = [conversion_factor(D, H) for D, H in zip(d, h)]
    factors = np.array([r[0] for r in rows])

    return pd.DataFrame({
        "diameter_mm": d,
        "height_mm": h,
        "fc_raw_MPa": fc0,
        "diameter_class_mm": [r[1] for r in rows],
        "H_over_D": [r[2] for r in rows],
        "factor_applied": factors,
        "fc_normalised_MPa": factors * fc0,
        "note": [r[3] for r in rows],
    })
