import numpy as np
import pandas as pd



# ----------------------------------------------------------
#                   NORMALIZE RAW FC INPUT
# ----------------------------------------------------------
# (diameter_mm, height_mm) : factor_to_multiply_raw_strength
TABLE_FACTORS = {
    (150, 300): 1.00,
    (100, 300): 1.05,
    (120, 320): 1.00 / 0.95,
    (100, 200): 0.92 / 0.95,
    (100, 100): 0.98 / 1.06,
    (100, 150): 0.96,
    (75, 150): 1.00 / 1.06,
    (75, 100): 0.91,
    (75,  75): 0.85,
    (50,  75): 0.91,
    (50,  50): 0.80,
}

TABLE_TOLERANCE = 5  # [mm] round-off 

# --------------------------------------------------------
#    CONTINUOUS INTERPOLATION 
# --------------------------------------------------------

# (a) aspect-ratio (L/D) factor – ASTM C42 values, linearly interpolated
_LD_POINTS  = np.array([1.00, 1.25, 1.50, 1.75, 2.00])
_LD_FACTORS = np.array([0.87, 0.93, 0.96, 0.98, 1.00])

def length_factor(L_over_D: np.ndarray) -> np.ndarray:
    """
    Bartlett & MacGregor (ASTM C42) length correction, linear-interp.
    Values above L/D = 2.0 are taken as 1.00 (no correction).
    """
    return np.interp(L_over_D, _LD_POINTS, _LD_FACTORS, left=_LD_FACTORS[0], right=1.00)


# (b) diameter factor – Bartlett & MacGregor size effect (in 'multiplicative' form)
_D_POINTS  = np.array([50,  75, 100, 150])  # mm
_D_FACTORS = np.array([0.92, 0.97, 1.00, 1.00])

def diameter_factor(D: np.ndarray) -> np.ndarray:
    """
    Size effect for drilled cores relative to a 150 mm cylinder.
    Linear-interp.  Diameters >150 mm are capped at 1.00 (no size effect).
    """
    return np.interp(D, _D_POINTS, _D_FACTORS, left=_D_FACTORS[0], right=1.00)


# -------------------------------------------------------
#    PUBLIC FUNCTION
# -------------------------------------------------------
def normalise_to_150x300(
    diam_mm: np.ndarray | list,
    height_mm: np.ndarray | list,
    fc_raw_mpa: np.ndarray | list,
    use_table_first: bool = True,
) -> pd.DataFrame:
    # Normalise user-supplied compressive strengths to the 150×300 mm reference.
    d   = np.asarray(diam_mm,   dtype=float)
    h   = np.asarray(height_mm, dtype=float)
    fc0 = np.asarray(fc_raw_mpa, dtype=float)

    if not (d.size == h.size == fc0.size):
        raise ValueError("diameter, height and fc arrays must be the same length")

    factors = np.empty_like(fc0, dtype=float)

    for i, (D, H) in enumerate(zip(d, h)):
        found_factor = None

        if use_table_first:
            # Look for a match in the hard-wired table within the tolerance
            for (Dt, Ht), Ft in TABLE_FACTORS.items():
                if abs(D - Dt) <= TABLE_TOLERANCE and abs(H - Ht) <= TABLE_TOLERANCE:
                    found_factor = Ft
                    break

        if found_factor is None:
            # ----  Continuous model  ----
            fld = length_factor(H / D)
            fD  = diameter_factor(D)
            found_factor = fld * fD

        factors[i] = found_factor

    fc_norm = factors * fc0

    return pd.DataFrame(
        {
            "diameter_mm": d,
            "height_mm":   h,
            "fc_raw_MPa":  fc0,
            "factor_applied": factors,
            "fc_normalised_MPa": fc_norm,
        }
    )