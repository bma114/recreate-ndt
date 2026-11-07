import logging
import os, json, gc
import joblib
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from .utils import normalise_to_150x300
from . import stats

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import catboost as cb
import traceback

import re  # Required for regex replacements
import environ

logger = logging.getLogger(__name__)

# Import environment variables
env = environ.Env()
environ.Env.read_env()


# Render the webpage using the index html 
def home(request):
        return render(request, 'index.html')


# AWS S3 Configuration
AWS_BUCKET_NAME = "compressive-strength-ndt"
# S3_MODEL_PREFIX = "models/"


# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=env("AWS_ACCESS_KEY"),
    aws_secret_access_key=env("AWS_SECRET_KEY")
)

SCALER_PATH   = "scalers/{ndt}_scaler.pkl"
MODEL_PATTERN = "Models/{ndt}_catboost_model_fold{fold}_cmp.pkl"

# Global in-memory cache
_models_cache = {}


# Function to load scaler from S3 (cached after the first request)
_scaler_cache = {}

def load_scaler(ndt):
    """Fetches the scaler from S3 and returns the loaded object."""
    if ndt in _scaler_cache:
        return _scaler_cache[ndt]
    file_stream = BytesIO()
    s3_client.download_fileobj(AWS_BUCKET_NAME, SCALER_PATH.format(ndt=ndt), file_stream)
    file_stream.seek(0)
    scaler = joblib.load(file_stream)

    # Verify the scaler is a valid instance of a scikit-learn scaler
    if not isinstance(scaler, (MinMaxScaler, StandardScaler)):
        raise TypeError(f"Scaler wrong type: {type(scaler)}")

    _scaler_cache[ndt] = scaler
    return scaler

# Function to load models from S3
def get_models(ndt):
    # if not yet loaded, fetch from S3 and cache
    if ndt not in _models_cache:
        # purge other NDTs
        for other in list(_models_cache):
            if other != ndt:
                del _models_cache[other]
        gc.collect()

        folds = []
        for i in range(1, 11):
            file_stream = BytesIO()
            key = MODEL_PATTERN.format(ndt=ndt, fold=i)
            s3_client.download_fileobj(AWS_BUCKET_NAME, key, file_stream)
            file_stream.seek(0)
            folds.append(joblib.load(file_stream))
        _models_cache[ndt] = folds

    return _models_cache[ndt]

feature_columns = {
    "upv": ["Country", "Specimen Type", "Specimen Age (days)", "Rebar Present", "UPV_Device_Brand", "Transducer Diameter (mm)", 
            "Transducer Frequency (kHz)", "UPV_Procedure", "Test_Type", "No. UPV Tests", "Vp", "Core Specimen", 
            "Height (mm)", "Width/Diameter (mm)", "Max Aggregate Size (mm)", "W/C Ratio", "Design Strength (MPa)"],
    "rh": ["Country", "Specimen Type", "Specimen Age (days)", "Rebar Present", "RH_Device_Brand", "RH_Procedure", 
           "Orientation", "No. RH Tests", "RN", "Core Specimen", "Height (mm)", "Width/Diameter (mm)", 
           "Max Aggregate Size (mm)", "W/C Ratio", "Design Strength (MPa)"],
    "sonreb": ["Country", "Specimen Type", "Specimen Age (days)", "Rebar Present", "UPV_Device_Brand", "Transducer Diameter (mm)", 
               "Transducer Frequency (kHz)", "UPV_Procedure", "Test_Type", "No. UPV Tests", "Vp", "RH_Device_Brand", 
               "RH_Procedure", "Orientation", "No. RH Tests", "RN", "Core Specimen", "Height (mm)", 
               "Width/Diameter (mm)", "Max Aggregate Size (mm)", "W/C Ratio", "Design Strength (MPa)"]
}

# Feature mapping: Matching frontend inputs to model feature names
feature_mapping = {
    "country": "Country",
    "specimen_type": "Specimen Type",
    "specimen_age": "Specimen Age (days)",
    "rebar_present": "Rebar Present",
    "upv_device_brand": "UPV_Device_Brand",
    "upv_transd_diam": "Transducer Diameter (mm)",
    "upv_transd_freq": "Transducer Frequency (kHz)",
    "upv_standard": "UPV_Procedure",
    "test_type": "Test_Type",
    "no_upv_tests": "No. UPV Tests",
    "velocity": "Vp",
    "rh_device_brand": "RH_Device_Brand",
    "rh_standard": "RH_Procedure",
    "orientation": "Orientation",
    "no_rh_tests": "No. RH Tests",
    "rebound_number": "RN",
    "concrete_specimen": "Core Specimen",
    "height": "Height (mm)",
    "width_diameter": "Width/Diameter (mm)",
    "max_agg": "Max Aggregate Size (mm)",
    "wc_ratio": "W/C Ratio",
    "design_strength": "Design Strength (MPa)"
}

categorical_features = {"Country", "Specimen Type", "Rebar Present", "UPV_Device_Brand", "UPV_Procedure",
                        "Test_Type", "RH_Device_Brand", "RH_Procedure", "Orientation", "Core Specimen"}

def preprocess_features(features, ndt, scaler):
    """Prepares input features for prediction."""
    feature_order = feature_columns[ndt]
    
    # Create a dictionary that maps the HTML `id` to the feature names expected by the model
    mapped_features = {}
    for feature_id, feature_value in features.items():
        if feature_id in feature_mapping:
            model_feature_name = feature_mapping[feature_id]
        elif feature_id in feature_order:
            # Already a model column name
            model_feature_name = feature_id
        else:
            # neither an HTML id nor a model column → ignore
            continue
        
        mapped_features[model_feature_name] = feature_value
    
    # # Debugging: Print out the mapped features to check for correctness

    # Convert to DataFrame with the correct column names (use mapped_features and feature_order)
    df = pd.DataFrame([mapped_features], columns=feature_order)

    # Separate categorical and numerical features
    categorical_cols = [col for col in feature_order if col in categorical_features]
    categorical_df = df[categorical_cols]  
    numerical_df = df.drop(columns=categorical_df.columns)

    # Normalize categorical features
    categorical_df = categorical_df.astype("string").fillna("missing")  # Handle missing values
    categorical_df = categorical_df.apply(lambda col: col.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True))

    # Debugging: Print out the DataFrame to check feature names, and entire categorical and numerical DataFrames
    
    # Apply transform to numerical data (we only scale numerical features)
    scaled_numerical = scaler.transform(numerical_df)
    
    # Combine categorical and scaled numerical features
    processed_df = pd.concat([pd.DataFrame(scaled_numerical, columns=numerical_df.columns), categorical_df], axis=1)
    
    return processed_df

def predict_strength(features, ndt, scaler):
    """Runs a prediction using all trained models and returns the average."""
    
    df = preprocess_features(features, ndt, scaler)

    # Determine the categorical feature indices using the same ordering as in training:
    cat_feature_indices = [df.columns.get_loc(col) for col in df.columns 
                           if col in categorical_features]

    # Build a Pool with these indices so catboost knows which columns are categorical
    data_pool = cb.Pool(df, cat_features=cat_feature_indices)

    # Lazy-load exactly the 10 models you need
    models = get_models(ndt)

    # Run each model and collect predictions
    preds = []
    for m in models:
        p = m.predict(data_pool)[0]
        preds.append(p)

    # Return the mean
    return float(np.mean(preds))



##----------- Predictions for Individual tab -----------##

# Hard limits for numeric inputs (physically sensible values); the form uses the same limits
INDIV_NUMERIC_LIMITS = {
    "specimen_age":    ("concrete age (days)", 1, 36500),
    "velocity":        ("pulse velocity (m/s)", 1000, 7000),
    "no_upv_tests":    ("number of UPV readings", 1, 100),
    "upv_transd_freq": ("transducer frequency (kHz)", 10, 500),
    "upv_transd_diam": ("transducer diameter (mm)", 10, 100),
    "rebound_number":  ("rebound number", 10, 100),
    "no_rh_tests":     ("number of impacts", 1, 200),
    "design_strength": ("design strength (MPa)", 5, 150),
    "max_agg":         ("maximum aggregate size (mm)", 2, 100),
    "wc_ratio":        ("water/cement ratio", 0.15, 1.5),
    "height":          ("specimen height (mm)", 20, 1000),
    "width_diameter":  ("specimen width or diameter (mm)", 20, 500),
}
INDIV_REQUIRED = {
    "common": ["country", "specimen_type", "specimen_age", "rebar_present"],
    "upv":    ["upv_device_brand", "upv_standard", "test_type", "no_upv_tests", "velocity"],
    "rh":     ["rh_device_brand", "rh_standard", "orientation", "no_rh_tests", "rebound_number"],
}


def validate_individual(ndt, features):
    """Return (clean_features, errors). Mirrors the checks made in the browser."""
    required = INDIV_REQUIRED["common"][:]
    if ndt in ("upv", "sonreb"):
        required += INDIV_REQUIRED["upv"]
    if ndt in ("rh", "sonreb"):
        required += INDIV_REQUIRED["rh"]

    clean, errors = {}, []
    for key, value in features.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key in INDIV_NUMERIC_LIMITS:
            label, lo, hi = INDIV_NUMERIC_LIMITS[key]
            try:
                x = float(value)
            except (TypeError, ValueError):
                errors.append(f"{label} must be a number")
                continue
            if not np.isfinite(x) or not lo <= x <= hi:
                errors.append(f"{label} must be between {lo:g} and {hi:g}")
                continue
            clean[key] = x
        else:
            clean[key] = str(value).strip()[:80]

    missing = [k for k in required if k not in clean]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    return clean, errors


@csrf_exempt
def predict_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Send a POST request with input data.'}, status=405)

    try:
        data = json.loads(request.body)
    except (ValueError, TypeError):
        return JsonResponse({"error": "The request could not be read."}, status=400)

    ndt = data.get("ndt")
    features = data.get("features", {})
    if ndt not in ("upv", "rh", "sonreb") or not isinstance(features, dict):
        return JsonResponse({"error": "Select a test method and fill in the form."}, status=400)

    features, errors = validate_individual(ndt, features)
    if errors:
        return JsonResponse({"error": "Please check the inputs: " + "; ".join(errors) + "."}, status=400)

    try:
        scaler = load_scaler(ndt)
        prediction = predict_strength(features, ndt, scaler)
    except Exception:  # model loading or prediction failure
        logger.exception("Individual prediction failed")
        return JsonResponse({"error": "The model could not make a prediction. Please try again."}, status=500)

    # Reverse the target transformation used in training
    if ndt in ("upv", "sonreb"):
        prediction = float(np.exp(prediction))
    else:
        prediction = float(np.square(prediction))

    return JsonResponse({
        "value": round(prediction, 2),
        "prediction": f"f<sub>c,cyl</sub> = {prediction:.1f} MPa",
    })



##----------- Shared input handling for the DT+NDT and DT-only tabs -----------##

MIN_CORES = 8          # minimum paired core tests (n)
MIN_EXTRA = 3          # minimum additional NDT locations (m)

# 5-95 % fractiles of f_c,cyl in the training databases (DRAFTv8, Section 3.1.5)
TRAINING_FC_RANGE = {"upv": (8.34, 61.7), "rh": (11.5, 84.3), "sonreb": (11.1, 54.8)}


class InputError(ValueError):
    """Raised for user input problems; returned to the browser as a 400."""


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def read_table(data, columns, label):
    """Read the parallel lists of a table, drop fully blank rows and reject
    partly filled or non-numeric rows. Returns {column: [floats]}."""
    lists = {c: _as_list(data.get(c)) for c in columns}
    n_rows = max((len(v) for v in lists.values()), default=0)
    out = {c: [] for c in columns}
    for i in range(n_rows):
        cells = [str(lists[c][i]).strip() if i < len(lists[c]) else "" for c in columns]
        if all(x == "" for x in cells):
            continue
        if any(x == "" for x in cells):
            raise InputError(f"{label}, row {i + 1}: please complete or delete this row.")
        try:
            vals = [float(x) for x in cells]
        except ValueError:
            raise InputError(f"{label}, row {i + 1}: non-numeric value.")
        if any(not np.isfinite(v) or v <= 0 for v in vals):
            raise InputError(f"{label}, row {i + 1}: values must be positive numbers.")
        for c, v in zip(columns, vals):
            out[c].append(v)
    return out


def normalisation_table(df_norm):
    return [
        {
            "D": float(r.diameter_mm), "H": float(r.height_mm),
            "class": int(r.diameter_class_mm), "H_over_D": float(r.H_over_D),
            "fc_core": float(r.fc_raw_MPa), "factor": float(r.factor_applied),
            "fc_cyl": float(r.fc_normalised_MPa), "note": r.note,
        }
        for r in df_norm.itertuples()
    ]


# Cached global database (plots only)
_global_db_cache = {}

def get_global_db(ndt):
    if "xlsx" not in _global_db_cache:
        obj = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key="data/Full Databases - Cleaned.xlsx")
        _global_db_cache["xlsx"] = pd.ExcelFile(BytesIO(obj['Body'].read()))
    sheet = {'upv': 'UPV', 'rh': 'Rebound Hammer', 'sonreb': 'SonReb'}[ndt]
    if sheet not in _global_db_cache:
        _global_db_cache[sheet] = _global_db_cache["xlsx"].parse(sheet_name=sheet)
    return _global_db_cache[sheet]


def _clean(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


##----------- Predictions for Combined DT + NDT Tab -----------##

NUMERIC_FEATS = {
    "Specimen Age (days)", "Transducer Diameter (mm)", "Transducer Frequency (kHz)",
    "No. UPV Tests", "No. RH Tests", "Vp", "RN", "Height (mm)", "Width/Diameter (mm)",
    "Max Aggregate Size (mm)", "W/C Ratio", "Design Strength (MPa)",
}


@csrf_exempt
def analyse_combined_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))

        ndt = _as_list(data.get('combined_ndt'))[0] if data.get('combined_ndt') else None
        if ndt not in ("upv", "rh", "sonreb"):
            raise InputError("Please select an NDT method.")
        has_upv = ndt in ("upv", "sonreb")
        has_rh = ndt in ("rh", "sonreb")

        # Card-level features (one value for the whole test region)
        card_feats = {}
        for html_id, model_name in feature_mapping.items():
            vals = _as_list(data.get(html_id))
            if not vals:
                continue
            v = str(vals[0]).strip()
            if v == "":
                continue
            if model_name in NUMERIC_FEATS:
                try:
                    x = float(v)
                except ValueError:
                    raise InputError(f"'{html_id}' must be a number.")
                if html_id in INDIV_NUMERIC_LIMITS:
                    label, lo, hi = INDIV_NUMERIC_LIMITS[html_id]
                    if not (np.isfinite(x) and lo <= x <= hi):
                        raise InputError(f"The {label} must be between {lo:g} and {hi:g}.")
                card_feats[model_name] = x
            else:
                card_feats[model_name] = v
        card_feats["Core Specimen"] = "Core"     # every compression specimen is a core

        # Calibration set (n): paired cores and NDT
        n_cols = (["velocity_n[]"] if has_upv else []) + (["rebound_number_n[]"] if has_rh else []) \
                 + ["width_diameter_n[]", "height_n[]", "fc_n[]"]
        n_tab = read_table(data, n_cols, "Calibration set")
        # Additional set (m): NDT only
        m_cols = (["velocity_m[]"] if has_upv else []) + (["rebound_number_m[]"] if has_rh else [])
        m_tab = read_table(data, m_cols, "Additional NDT set")

        # Hard limits on the table values (the form applies the same limits)
        limits = {"velocity_n[]": ("pulse velocity", 1000, 7000), "velocity_m[]": ("pulse velocity", 1000, 7000),
                  "rebound_number_n[]": ("rebound number", 10, 100), "rebound_number_m[]": ("rebound number", 10, 100),
                  "width_diameter_n[]": ("core diameter (mm)", 20, 300), "height_n[]": ("core height (mm)", 20, 600),
                  "fc_n[]": ("core strength (MPa)", 1, 200)}
        for tab, label in ((n_tab, "Calibration set"), (m_tab, "Additional NDT set")):
            for col, vals in tab.items():
                name, lo, hi = limits[col]
                for i, v in enumerate(vals):
                    if not lo <= v <= hi:
                        raise InputError(f"{label}, row {i + 1}: the {name} must be between {lo:g} and {hi:g}.")

        n = len(n_tab["fc_n[]"])
        m = len(m_tab[m_cols[0]])
        if n < MIN_CORES:
            raise InputError(f"At least {MIN_CORES} paired core tests are needed (got {n}).")
        if m < MIN_EXTRA:
            raise InputError(f"At least {MIN_EXTRA} additional NDT results are needed (got {m}).")

        # Normalise core strengths to the 150 x 300 mm cylinder
        df_norm = normalise_to_150x300(n_tab["width_diameter_n[]"], n_tab["height_n[]"], n_tab["fc_n[]"])
        fc_n = df_norm["fc_normalised_MPa"].values

        # Model inputs for each location
        def row_dict(vp, rn, h, d):
            r = dict(card_feats)
            if has_upv:
                r["Vp"] = vp
            if has_rh:
                r["RN"] = rn
            r["Height (mm)"] = h
            r["Width/Diameter (mm)"] = d
            return r

        vp_n = n_tab.get("velocity_n[]", [None] * n)
        rn_n = n_tab.get("rebound_number_n[]", [None] * n)
        vp_m = m_tab.get("velocity_m[]", [None] * m)
        rn_m = m_tab.get("rebound_number_m[]", [None] * m)
        # The additional locations have no core: use the mean core geometry
        h_avg = float(np.mean(n_tab["height_n[]"]))
        d_avg = float(np.mean(n_tab["width_diameter_n[]"]))
        n_rows = [row_dict(vp_n[i], rn_n[i], n_tab["height_n[]"][i], n_tab["width_diameter_n[]"][i]) for i in range(n)]
        m_rows = [row_dict(vp_m[i], rn_m[i], h_avg, d_avg) for i in range(m)]

        # GLOBAL calibration: ensemble of the 10 CatBoost models
        scaler = load_scaler(ndt)
        models = get_models(ndt)
        X_n = pd.concat([preprocess_features(r, ndt, scaler) for r in n_rows], ignore_index=True)
        X_m = pd.concat([preprocess_features(r, ndt, scaler) for r in m_rows], ignore_index=True)

        # Average the 10 models in their training space, then back-transform
        # (same as the Individual tab and the research code)
        raw_n = np.mean([mdl.predict(X_n) for mdl in models], axis=0)
        raw_m = np.mean([mdl.predict(X_m) for mdl in models], axis=0)
        if ndt in ("upv", "sonreb"):              # trained on ln(fc)
            pred_n, pred_m = np.exp(raw_n), np.exp(raw_m)
        else:                                      # RH trained on sqrt(fc)
            pred_n, pred_m = np.square(raw_n), np.square(raw_m)

        glob = stats.calibrated(fc_n, np.log(pred_n), np.log(pred_m), p=1, shift=True)
        glob["pred_n_unshifted"] = pred_n.tolist()
        glob["pred_m_unshifted"] = pred_m.tolist()

        # LOCAL calibration: log-log least squares, ln f = a + b ln Vp (+ c ln RN)
        def design(vp, rn, k):
            cols = [np.ones(k)]
            if has_upv:
                cols.append(np.log(vp))
            if has_rh:
                cols.append(np.log(rn))
            return np.column_stack(cols)

        A_n = design(np.array(vp_n, dtype=float), np.array(rn_n, dtype=float), n)
        A_m = design(np.array(vp_m, dtype=float), np.array(rn_m, dtype=float), m)
        y = np.log(fc_n)
        p_loc = A_n.shape[1]
        if np.linalg.matrix_rank(A_n) < p_loc:
            raise InputError("Local calibration needs NDT values that vary between the cores.")
        coef, *_ = np.linalg.lstsq(A_n, y, rcond=None)
        ln_fit_n, ln_fit_m = A_n @ coef, A_m @ coef
        ss_res = float(np.sum((y - ln_fit_n) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        loc = stats.calibrated(fc_n, ln_fit_n, ln_fit_m, p=p_loc, shift=False)
        names = ["a"] + (["b_Vp"] if has_upv else []) + (["c_RN" if has_upv else "b_RN"] if has_rh else [])
        loc["coefficients"] = {k: float(v) for k, v in zip(names, coef)}
        loc["R2_log"] = r2

        # Local plot: data and fitted curve (other variable at its geometric mean)
        def curve(var):
            vals = np.array((vp_n + vp_m) if var == "Vp" else (rn_n + rn_m), dtype=float)
            xs = np.linspace(vals.min(), vals.max(), 50)
            ln_y = coef[0]
            if has_upv:
                vp = xs if var == "Vp" else np.exp(np.mean(np.log(vp_n)))
                ln_y = ln_y + coef[1] * np.log(vp)
            if has_rh:
                rn = xs if var == "RN" else np.exp(np.mean(np.log(rn_n)))
                ln_y = ln_y + coef[-1] * np.log(rn)
            return {"x": xs.tolist(), "y": np.exp(np.broadcast_to(ln_y, xs.shape)).tolist()}

        local_plots = {
            "Xn_Vp": vp_n if has_upv else [],
            "Xn_RN": rn_n if has_rh else [],
            "Y": fc_n.tolist(),
            "curve_Vp": curve("Vp") if has_upv else None,
            "curve_RN": curve("RN") if has_rh else None,
        }

        # Global database for the plots
        df_global = get_global_db(ndt)
        if ndt in ("upv", "rh"):
            global_plots = {"X_global": df_global["Vp" if ndt == "upv" else "RN"].tolist(),
                            "Y_global": df_global["fc,cyl"].tolist()}
        else:
            global_plots = {"X_global_Vp": df_global["Vp"].tolist(),
                            "X_global_RN": df_global["RN"].tolist(),
                            "Y_global": df_global["fc,cyl"].tolist()}

        # Warnings
        warnings_out = []
        lo, hi = TRAINING_FC_RANGE[ndt]
        if not (lo <= fc_n.mean() <= hi):
            warnings_out.append(
                f"The mean normalised core strength ({fc_n.mean():.1f} MPa) is outside the 5-95% range "
                f"of the {ndt.upper()} training data ({lo}-{hi} MPa). Treat the global calibration with caution.")
        for i, note in enumerate(df_norm["note"]):
            if note:
                warnings_out.append(f"Core {i + 1}: {note}.")

        return JsonResponse(_clean({
            "ndt": ndt,
            "normalisation": normalisation_table(df_norm),
            "global": glob,
            "local": loc,
            "dt_only": glob["dt_only"],
            "local_plots": local_plots,
            "global_plots": global_plots,
            "warnings": warnings_out,
        }))

    except InputError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


##-------------- Predictions for DT Only Tab --------------##

@csrf_exempt
def analyse_dt_only_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        tab = read_table(data, ["diam[]", "height[]", "fc[]"], "Core table")
        n = len(tab["fc[]"])
        if n < MIN_CORES:
            raise InputError(f"At least {MIN_CORES} core tests are needed (got {n}).")

        df_norm = normalise_to_150x300(tab["diam[]"], tab["height[]"], tab["fc[]"])
        dt = stats.dt_only(df_norm["fc_normalised_MPa"].values)

        warnings_out = [f"Core {i + 1}: {note}." for i, note in enumerate(df_norm["note"]) if note]

        return JsonResponse(_clean({
            "dt_only": dt,
            "normalisation": normalisation_table(df_norm),
            "warnings": warnings_out,
        }))

    except InputError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
