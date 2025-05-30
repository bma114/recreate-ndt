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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
import catboost as cb
import traceback

import re  # Required for regex replacements
import environ

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


# Function to load scaler from S3
def load_scaler(ndt):
    """Fetches the scaler from S3 and returns the loaded object."""    
    file_stream = BytesIO()
    s3_client.download_fileobj(AWS_BUCKET_NAME, SCALER_PATH.format(ndt=ndt), file_stream)
    file_stream.seek(0)
    scaler = joblib.load(file_stream)

    # Verify the scaler is a valid instance of a scikit-learn scaler
    if not isinstance(scaler, (MinMaxScaler, StandardScaler)):
        raise TypeError(f"Scaler wrong type: {type(scaler)}")
    
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
    print(f"Mapped Features: {mapped_features}")

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
    print(f"Feature DataFrame:\n{df.to_string(index=False)}")  # Print entire DataFrame
    print(f"\nCategorical Data:\n{categorical_df.to_string(index=False)}")
    print(f"\nNumerical Data:\n{numerical_df.to_string(index=False)}")
    
    # Apply transform to numerical data (we only scale numerical features)
    print(f"Numerical Data before scaling:\n{numerical_df.to_string(index=False)}")
    scaled_numerical = scaler.transform(numerical_df)
    print(f"Scaled Numerical Data:\n{scaled_numerical}")
    
    # Combine categorical and scaled numerical features
    processed_df = pd.concat([pd.DataFrame(scaled_numerical, columns=numerical_df.columns), categorical_df], axis=1)
    
    return processed_df

def predict_strength(features, ndt, scaler):
    """Runs a prediction using all trained models and returns the average."""
    
    df = preprocess_features(features, ndt, scaler)
    print(f"Input to model: {df}")  # Check what is being passed to the model

    # Determine the categorical feature indices using the same ordering as in training:
    cat_feature_indices = [df.columns.get_loc(col) for col in df.columns 
                           if col in categorical_features]
    print(f"Categorical feature indices: {cat_feature_indices}")

    # Build a Pool with these indices so catboost knows which columns are categorical
    data_pool = cb.Pool(df, cat_features=cat_feature_indices)

    # Lazy-load exactly the 10 models you need
    models = get_models(ndt)

    # Run each model and collect predictions
    preds = []
    for m in models:
        p = m.predict(data_pool)[0]
        preds.append(p)
    print(f"Raw predictions: {preds}")

    # Return the mean
    return float(np.mean(preds))



##----------- Predictions for Indiviudal tab -----------##

@csrf_exempt  # Disable CSRF for simplicity in testing
def predict_view(request):
    if request.method != 'POST':
        return JsonResponse({'message':'Send a POST request with input data'}, status=405)
    
    try:
        # Parse JSON data from request
        data = json.loads(request.body)

        ndt = data.get("ndt")
        features = data.get("features", {})

        if ndt not in ["upv","rh","sonreb"] or not isinstance(features, dict):
            return JsonResponse({"error":"Invalid request"}, status=400)
        
        # Load the correct scaler from S3
        scaler = load_scaler(ndt)
        
        # Make prediction based on correct models
        prediction = predict_strength(features, ndt, scaler)

        # Reverse the transformation
        if ndt in ("upv", "sonreb"):
            prediction = np.exp(prediction) # Reverse the log transformation
        elif ndt == "rh":
            prediction = np.square(prediction) # Reverse the square root transformation

        prediction_result = f"f<sub>c,cyl</sub> = {prediction:.2f} MPa"
        print(f"Prediction Result: {prediction_result}")  # Debugging: Print prediction result

        # Returning HTML formatted result
        return JsonResponse({'prediction': prediction_result})

    except Exception as e:
        print(f"Error: {e}")  # Debugging: Print error message
        return JsonResponse({'error': str(e)}, status=400)
    


##----------- Predictions for Combined DT + NDT Tab -----------##

@csrf_exempt
def analyse_combined_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))


        ndt = data['combined_ndt'][0]  # 'upv','rh' or 'sonreb'
        print("🔧 Selected NDT:", ndt)

        # card-level features: pm;y non-empty scalar card features
        card_feats = {}
        for html_id, model_name in feature_mapping.items():
            if html_id in data:
                vals = data[html_id]
                if isinstance(vals, list):
                    v = vals[0].strip()
                else:
                    v = str(vals).strip()

                # Skip if blank or missing
                if v == "":
                    continue

                # Convert numeric inputs to float:
                numeric_feats = {
                    "Specimen Age (days)", "Transducer Diameter (mm)",
                    "Transducer Frequency (kHz)", "No. UPV Tests",
                    "Vp", "RN", "Height (mm)", "Width/Diameter (mm)",
                    "Max Aggregate Size (mm)", "W/C Ratio", "Design Strength (MPa)"
                }
                if model_name in numeric_feats:
                    card_feats[model_name] = float(v)
                else:
                    card_feats[model_name] = v

        # Every compression specimen is assumed to be a core
        card_feats["Core Specimen"] = "Core"

        # Normalize compression strength for n-set
        diam = data["width_diameter_n[]"]
        height = data["height_n[]"]
        raw_fc = data["fc_n[]"]
        df_norm = normalise_to_150x300(diam, height, raw_fc)

        # Build calibration dataframe (n_set)
        n_rows = len(data['fc_n[]'])
        n_dict = {}
        # For UPV / RH columns:
        if ndt in ('upv','sonreb'):
            n_dict['Vp']               = [ float(x) for x in data['velocity_n[]'] ]
        if ndt in ('rh','sonreb'):
            n_dict['RN']               = [ float(x) for x in data['rebound_number_n[]'] ]
        # Core dimensions and fc
        n_dict['Width/Diameter (mm)']  = [ float(x) for x in data['width_diameter_n[]'] ]
        n_dict['Height (mm)']          = [ float(x) for x in data['height_n[]'] ]
        n_dict['fc,cyl (MPa)']         = df_norm["fc_normalised_MPa"].tolist()

        n_df = pd.DataFrame(n_dict)
        n_df = n_df.rename(columns={
            'Width/Diameter (mm)': 'width_diameter',
            'Height (mm)': 'height',
        })


        # Duplicate card_feats across n rows
        for k,v in card_feats.items():
            n_df[k] = v

        # Complementary dataframe (m_set)
        # m_rows = len(data['velocity_m[]'] or data['rebound_number_m[]'])
        m_dict = {}
        if ndt in ('upv','sonreb'):
            m_dict['Vp']               = [ float(x) for x in data['velocity_m[]'] ]
        if ndt in ('rh','sonreb'):
            m_dict['RN']               = [ float(x) for x in data['rebound_number_m[]'] ]

        m_df = pd.DataFrame(m_dict)

        # Propagate average core dimensions into complementary set
        avg_height = n_df['height'].mean()
        avg_width  = n_df['width_diameter'].mean()

        # assign to every row
        m_df['height']          = [avg_height] * len(m_df)
        m_df['width_diameter']  = [avg_width] * len(m_df)

        for k,v in card_feats.items():
            m_df[k] = v

        # GLOBAL calibration via CatBoost
        # preprocess_features expects a single-row dict, so map each row:
        scaler = load_scaler(ndt)
        models  = get_models(ndt)

        # Build and preprocess each row individually, then concat:
        n_dicts = []
        for _, row in n_df.iterrows():
            d = row.to_dict()                      
            d.update(card_feats)                   # now contains all the card inputs
            n_dicts.append(d)

        # preprocess each
        processed_n = [ preprocess_features(d, ndt, scaler) for d in n_dicts ]
        X_n = pd.concat(processed_n, ignore_index=True)

        m_dicts = []
        for _, row in m_df.iterrows():
            d = row.to_dict()                      
            d.update(card_feats)                   # now contains all the card inputs
            m_dicts.append(d)

        # preprocess each
        processed_m = [ preprocess_features(d, ndt, scaler) for d in m_dicts ]
        X_m = pd.concat(processed_m, ignore_index=True)

        # get ensemble predictions, reversing the training transform:
        n_preds, m_preds = [], []
        for mdl in models:
            raw_n = mdl.predict(X_n)
            raw_m = mdl.predict(X_m)
            if ndt in ("upv", "sonreb"):
                n_preds.append(np.exp(raw_n))
                m_preds.append(np.exp(raw_m))
            else:  
                n_preds.append(np.square(raw_n))
                m_preds.append(np.square(raw_m))
        n_pred = np.mean(n_preds, axis=0)
        m_pred = np.mean(m_preds, axis=0)

        # Statistical analysis:
        n = len(n_pred)
        m = len(m_pred)
        fc_is_n = n_df['fc,cyl (MPa)'].values

        # Mean value of 'fc_is' in subset n
        fcm_is = np.mean(fc_is_n)
        
        # Mean of predictions in subset m
        fcm_m_is_global = np.mean(m_pred)
        fcm_m_is_log_global = np.mean(np.log(m_pred))

        # Standard deviation of test set m_pred
        sfc_is_test_global = np.sqrt(np.sum((m_pred - fcm_m_is_global) ** 2) / (m - 1))
        sfc_is_test_log_global = np.sqrt(np.sum((np.log(m_pred) - fcm_m_is_log_global) ** 2) / (m - 1))
        
        # Standard deviation of the model prediction for subset n
        s_mod_test_global = np.sqrt(np.sum((fc_is_n - n_pred) ** 2) / (n - 2))
        s_mod_test_log_global = np.sqrt(np.sum((np.log(fc_is_n) - np.log(n_pred)) ** 2) / (n - 2))
        
        # Overall standard deviation
        sfc_is_global = np.sqrt(sfc_is_test_global ** 2 + s_mod_test_global ** 2)
        sfc_is_log_global = np.sqrt(sfc_is_test_log_global ** 2 + s_mod_test_log_global ** 2)

        # Calculate effective degree of freedom of total population
        n_eff_global = round(((sfc_is_test_global**2 + s_mod_test_global**2)**2)/((s_mod_test_global**4 / (n - 2)) + (sfc_is_test_global**4 / (m - 1))))
        k_neff_global = 1.71988 + 0.935204 * np.exp(-0.151838 * (n_eff_global + 1))
        k_dneff_global = 3.337755 + (19313890 - 3.337755)/(1 + ((n_eff_global + 1) / 0.005167562)**2.209513)
        
        # Overall coefficient of variation
        Vfc_is_log_global = np.sqrt(np.exp(sfc_is_log_global ** 2) - 1)
        Vfc_is_corr_global = Vfc_is_log_global * (k_dneff_global / (0.8 * 3.8))
       
        # Charactersitic strengths (should be the same)
        fck_is_global = fcm_m_is_global - k_neff_global * sfc_is_global                      # For normal distribution
        fck_is_log_global = np.exp(fcm_m_is_log_global - k_neff_global * sfc_is_log_global)  # For lognormal distribution

        bias_norm_global = fcm_m_is_global / fck_is_global              # If Normal distribution is assumed
        bias_log_global = np.exp(k_neff_global * Vfc_is_log_global)     # If Lognormal distribution is assumed

        # LOCAL calibration
        if ndt == "sonreb":
            Xn_lr = n_df[['Vp', 'RN']]
            Xm_lr = m_df[['Vp', 'RN']]
        elif ndt == "upv":
            Xn_lr = n_df[['Vp']]; Xm_lr = m_df[['Vp']]
        else:
            Xn_lr = n_df[['RN']]; Xm_lr = m_df[['RN']]

        # Single arrays for plotting
        if ndt == "sonreb":
            vp_vals = Xn_lr["Vp"].tolist()
            rn_vals = Xn_lr["RN"].tolist()
        else:
            vp_vals = []
            rn_vals = []
            if ndt == "upv":
                vp_vals = Xn_lr.values.flatten().tolist()
            else: 
                rn_vals = Xn_lr.values.flatten().tolist()
        
        lr = LinearRegression().fit(Xn_lr, fc_is_n)
        r2 = float(lr.score(Xn_lr, fc_is_n))

        n_pred_lr = lr.predict(Xn_lr)
        m_pred_lr = lr.predict(Xm_lr)

        # LOCAL Statistical analysis:        
        # Mean of predictions in subset m
        fcm_m_is_local = np.mean(m_pred_lr)
        fcm_m_is_log_local = np.mean(np.log(m_pred_lr))

        # Standard deviation of test set m_pred
        sfc_is_test_local = np.sqrt(np.sum((m_pred_lr - fcm_m_is_local) ** 2) / (m - 1))
        sfc_is_test_log_local = np.sqrt(np.sum((np.log(m_pred_lr) - fcm_m_is_log_local) ** 2) / (m - 1))
        
        # Standard deviation of the model prediction for subset n
        s_mod_test_local = np.sqrt(np.sum((fc_is_n - n_pred_lr) ** 2) / (n - 2))
        s_mod_test_log_local = np.sqrt(np.sum((np.log(fc_is_n) - np.log(n_pred_lr)) ** 2) / (n - 2))
        
        # Overall standard deviation
        sfc_is_local = np.sqrt(sfc_is_test_local ** 2 + s_mod_test_local ** 2)
        sfc_is_log_local = np.sqrt(sfc_is_test_log_local ** 2 + s_mod_test_log_local ** 2)
        
        # Calculate effective degree of freedom of total population
        n_eff_local = round(((sfc_is_test_local**2 + s_mod_test_local**2)**2)/((s_mod_test_local**4 / (n - 2)) + (sfc_is_test_local**4 / (m - 1))))
        k_neff_local = 1.71988 + 0.935204 * np.exp(-0.151838 * (n_eff_local + 1))
        k_dneff_local = 3.337755 + (19313890 - 3.337755)/(1 + (n_eff_local / 0.005167562)**2.209513)

        # Coefficient of variation
        Vfc_is_log_local = np.sqrt(np.exp(sfc_is_log_local ** 2) - 1)
        Vfc_is_corr_local = Vfc_is_log_local * (k_dneff_local / (0.8 * 3.8))
        
        # Charactersitic strengths (should be the same)
        fck_is_local = fcm_m_is_local - k_neff_local * sfc_is_local                      # For normal distribution
        fck_is_log_local = np.exp(fcm_m_is_log_local - k_neff_local * sfc_is_log_local)  # For lognormal distribution

        bias_norm_local = fcm_m_is_local / fck_is_local              # If Normal distribution is assumed
        bias_log_local = np.exp(k_neff_local * Vfc_is_log_local)     # If Lognormal distribution is assumed

        # Pull Excel data for plotting
        obj = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key="data/Full Databases - Cleaned.xlsx")
        xlsx = pd.ExcelFile(BytesIO(obj['Body'].read()))

        sheet_map = {'upv': 'UPV', 'rh': 'Rebound Hammer', 'sonreb': 'SonReb'}
        sheet = sheet_map[ndt]
        df_global = xlsx.parse(sheet_name=sheet)


        # build the summary dictionaries
        global_summary = {
            "f<sub>cm,n,is</sub>": fcm_is,                    # Mean of the calibration set
            "f<sub>cm,m,is</sub>": fcm_m_is_global,           # Mean of the complementary set
            "s<sub>fc,is</sub>": sfc_is_global,               # Overall std dev (Normal)
            "V<sub>fc</sub>": Vfc_is_log_global,              # CoV (lognormal)
            "f<sub>ck,is</sub>": float(fck_is_log_global),    # Characteristic strength (lognormal)
        }
        local_summary = {
            "f<sub>cm,n,is</sub>": fcm_is,                   # Mean of the calibration set
            "f<sub>cm,m,is</sub>": fcm_m_is_local,           # Mean of the complementary set
            "s<sub>fc,is</sub>": sfc_is_local,               # Overall std dev (Normal)
            "V<sub>fc</sub>": Vfc_is_log_local,              # CoV (lognormal)
            "f<sub>ck,is</sub>": float(fck_is_log_local),    # Characteristic strength (lognormal)
        }

        # build detail dictionaries
        global_details = {
            "f<sub>cm,n,is</sub>": fcm_is,                    # Mean of the calibration set
            "f<sub>cm,m,is</sub>": fcm_m_is_global,           # Mean of the complementary set (Normal)
            "s<sub>fc,is,test</sub>": sfc_is_test_global,     # Test variation  (Normal)
            "s<sub>theta,test</sub>": s_mod_test_global,      # Model variation (Normal)
            "s<sub>fc,is</sub>": sfc_is_global,               # Overall std dev (Normal)
            "V<sub>fc</sub>": Vfc_is_log_global,              # CoV (lognormal)
            "V<sub>fc,is,corr</sub>": Vfc_is_corr_global,     # CoV (log-Student-t)
            "n<sub>eff</sub>": n_eff_global,                  # DOF
            "k<sub>n,eff</sub>": k_neff_global,               # Sample size factor
            "f<sub>ck,is</sub>": float(fck_is_log_global),    # Characteristic strength (lognormal)
            "n_pred": n_pred.tolist(),                        # Calibration set predictions
            "m_pred": m_pred.tolist(),                        # Complementary set predictions
        }
        local_details = {
            "f<sub>cm,n,is</sub>": fcm_is,                   # Mean of the calibration set
            "f<sub>cm,m,is</sub>": fcm_m_is_local,           # Mean of the complementary set (Normal)
            "s<sub>fc,is,test</sub>": sfc_is_test_local,     # Test variation  (Normal)
            "s<sub>theta,test</sub>": s_mod_test_local,      # Model variation (Normal)
            "s<sub>fc,is</sub>": sfc_is_local,               # Overall std dev (Normal)
            "V<sub>fc</sub>": Vfc_is_log_local,              # CoV (lognormal)
            "V<sub>fc,is,corr</sub>": Vfc_is_corr_local,     # CoV (log-Student-t)
            "n<sub>eff</sub>": n_eff_local,                  # DOF
            "k<sub>n,eff</sub>": k_neff_local,               # Sample size factor
            "f<sub>ck,is</sub>": float(fck_is_log_local),    # Characteristic strength (lognormal)
            "n_pred": n_pred_lr.tolist(),                    # Calibration set predictions
            "m_pred": m_pred_lr.tolist(),                    # Complementary set predictions
        }

        if ndt == "sonreb":
            # turn the numpy array into a list of floats
            slope_vals = [float(c) for c in lr.coef_.tolist()]
            local_details["Slope"] = slope_vals
        else:
            local_details["Slope"] = float(lr.coef_[0])

        local_details["Intercept"] = float(lr.intercept_)
        local_details["R^2"]       = r2

        # Build plot dictionaries
        local_plots = {
            "Xn_Vp": vp_vals,
            "Xn_RN": rn_vals,
            "Y":     fc_is_n.tolist()
        }

        if ndt in ("upv", "rh"):
            global_x = df_global["Vp" if ndt=="upv" else "RN"].tolist()
            global_plots = {
                "X_global": global_x,
                "Y_global": df_global["fc,cyl"].tolist()
            }
        else:
            global_plots = {
                "X_global_Vp": df_global["Vp"].tolist(),
                "X_global_RN": df_global["RN"].tolist(),
                "Y_global":    df_global["fc,cyl"].tolist()
            }

        print("✅ About to return JSON response")
        return JsonResponse({
            "global_summary":  global_summary,
            "local_summary":   local_summary,
            "global_details":  global_details,
            "local_details":   local_details,
            "local_plots":     local_plots,
            "global_plots":    global_plots,
        })


    except KeyError as e:
        print("❌ Missing key:", e)
        traceback.print_exc()
        return JsonResponse({'error': f'Missing field: {e}'}, status=400)

    except json.JSONDecodeError as e:
        print("❌ JSON decode failed:", e)
        traceback.print_exc()
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    except Exception as e:
        print("🔥 Unexpected exception in analyse_combined_view:", e)
        traceback.print_exc()         # <— this prints the full stacktrace
        return JsonResponse({'error': str(e)}, status=500)



##-------------- Predictions for DT Only Tab --------------##

@csrf_exempt
def analyse_dt_only_view(request):
    print("🔥 analyse_dt_only_view called, method=", request.method)
    if request.method != 'POST':
        return JsonResponse({'error':'POST required'}, status=405)

    #  Parse JSON
    try:
        body = request.body.decode('utf-8')
        data = json.loads(body)
        print("✅ Parsed JSON keys:", data.keys())

    except Exception as e:
        print("❌ JSON parsing error:", e)
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    # Extract data arrays
    try:
        diam = list(map(float, data.get('diam[]', [])))
        ht   = list(map(float, data.get('height[]', [])))
        fc_raw   = list(map(float, data.get('fc[]', [])))
    except KeyError as e:
        return JsonResponse({'error': f'Missing field: {e}'}, status=400)  
    except ValueError as e:
        return JsonResponse({'error': 'Non-numeric value in input'}, status=400)

    # Sample size
    n = len(fc_raw)

    # Add minimum requirement
    if len(diam) < 8 or len(ht) < 8 or len(fc_raw) < 8:
        return JsonResponse({'error':'Please fill at least 8 rows.'}, status=400)
    
    # Normalize the results
    df_norm = normalise_to_150x300(diam, ht, fc_raw)
    print("Normalisation Parameters:", df_norm)

    # DT ONLY Statistical analysis:
    fc_is_n = df_norm['fc_normalised_MPa'].values

    # Sample Mean
    fcm_is = np.mean(fc_is_n)
    fcm_is_log = np.mean(np.log(fc_is_n))

    # Standard deviation
    sfc_is = np.sqrt(np.sum((fc_is_n - fcm_is) ** 2) / (n - 1))                 
    sfc_is_log = np.sqrt(np.sum((np.log(fc_is_n) - fcm_is_log) ** 2) / (n - 1))
    
    # Sample size factor
    k_n = 1.71988 + 0.935204 * np.exp(-0.151838 * (n))

    # Coefficient of variation
    Vfc_is = sfc_is / fcm_is                              
    Vfc_is_log = np.sqrt(np.exp(sfc_is_log ** 2) - 1) 
    Vfc_is_corr = Vfc_is_log * (k_n / (0.8 * 3.8))
    
    # Charactersitic strength
    fck_is = fcm_is * np.exp(-k_n * Vfc_is_log) 

    # Bias factor
    bias_logn = np.exp(k_n * Vfc_is_log)

    dt_summary = {
        "Mean fc (MPa)":      f"{fcm_is:.2f}",
        "Std Dev (MPa)":      f"{sfc_is:.2f}",
        "V_fc,is":            f"{Vfc_is_log:.3f}",
        "V_fc,is,corr":       f"{Vfc_is_corr:.3f}",
        "n":                  f"{n}",
        "k_n":                f"{k_n:.3f}",
        "Characteristic fck": f"{fck_is:.3f}",
        "Bias":               f"{bias_logn:.3f}",
    }

    print("✅ DT-only summary:", dt_summary)
    return JsonResponse({
        "dt_summary": dt_summary,
    })


# except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)


        
