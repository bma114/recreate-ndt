import os, json, gc
import joblib
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import catboost as cb

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


@csrf_exempt  # Disable CSRF for simplicity in testing
def predict_view(request):
    if request.method != 'POST':
        return JsonResponse({'message':'Send a POST request with input data'}, status=405)
    
    try:
        # Parse JSON data from request
        data = json.loads(request.body)
        # print("Received data in backend:", data)  # Debugging: Print raw data

        ndt = data.get("ndt")
        features = data.get("features", {})

        # print(f"Received Features:", features) # Debugging: Print features

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