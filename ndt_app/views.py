import os
import joblib
import numpy as np
import pandas as pd
import boto3
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
S3_MODEL_PREFIX = "models/"


# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=env("AWS_ACCESS_KEY"),
    aws_secret_access_key=env("AWS_SECRET_KEY")
)


# Load all models from S3
ndt_types = ["upv", "rh", "sonreb"]  # Example NDT types
models = {ndt: [] for ndt in ndt_types}


for ndt in ndt_types:
    for i in range(1, 6):  # Load 5 models per NDT
        model_key = f"Models/{ndt}_catboost_model_fold{i}.pkl"
        response = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=model_key)
        model = joblib.load(BytesIO(response['Body'].read()))
        models[ndt].append(model)


# Function to load scaler from S3
def load_scaler_from_s3(ndt_type):
    """Fetches the scaler from S3 and returns the loaded object."""
    scaler_key = f"scalers/{ndt_type}_scaler.pkl"  # Assuming the scalers are stored in a "Scalers" folder in S3
    
    file_stream = BytesIO()
    s3_client.download_fileobj(AWS_BUCKET_NAME, scaler_key, file_stream)

    file_stream.seek(0)
    scaler = joblib.load(file_stream)

    # Verify the scaler is a valid instance of a scikit-learn scaler
    if not isinstance(scaler, (MinMaxScaler, StandardScaler)):
        raise TypeError(f"Scaler is not of type 'MinMaxScaler' or 'StandardScaler', but is of type {type(scaler)}")
    
    return scaler

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
    
    # Convert to DataFrame with the correct column names (use mapped_features and feature_order)
    df = pd.DataFrame([mapped_features], columns=feature_order)

    # Separate categorical and numerical features
    categorical_cols = list(categorical_features.intersection(feature_order))
    categorical_df = df[categorical_cols]  
    numerical_df = df.drop(columns=categorical_df.columns)

    # Normalize categorical features
    categorical_df = categorical_df.astype("string").fillna("missing")  # Handle missing values
    categorical_df = categorical_df.apply(lambda col: col.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True))

    # Ensure that categorical features are not passed through the scaler, just the numerical ones
    # Apply transform to numerical data (we only scale numerical features)
    scaled_numerical = scaler.transform(numerical_df)
    
    # Combine categorical and scaled numerical features
    processed_df = pd.concat([pd.DataFrame(scaled_numerical, columns=numerical_df.columns), categorical_df], axis=1)
    
    return processed_df

def predict_strength(features, ndt, scaler):
    """Runs a prediction using all trained models and returns the average."""
    if ndt not in models:
        return JsonResponse({"error": f"Invalid NDT type: {ndt}"}, status=400)  # Ensure the NDT type is valid
    
    processed_features = preprocess_features(features, ndt, scaler)
    predictions = [model.predict(processed_features)[0] for model in models[ndt]]
    return float(np.mean(predictions))


# @csrf_exempt  # Disable CSRF for simplicity in testing
def predict_view(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from request
            data = json.loads(request.body)
            print("Received data in backend:", data)  # Debugging

            ndt_type = data.get("ndt")
            features = data.get("features", {})

            print(f"NDT type: {ndt_type}")  
            print(f"Received Features:", features)

            if not features:
                return JsonResponse({"error": "Features are missing!"}, status=400)
            
            # Check if NDT type is valid
            if ndt_type not in models:
                return JsonResponse({"error": "Invalid NDT type selected."}, status=400)

            # Load the correct scaler from S3
            scaler = load_scaler_from_s3(ndt_type)

            # Extract and preprocess features
            if not isinstance(features, dict):
                return JsonResponse({"error": "Features should be a dictionary."}, status=400)
            
            # Make prediction based on correct models
            prediction = predict_strength(features, ndt_type, scaler)

            # Reverse the transformation
            if ndt_type == "upv" or ndt_type == "sonreb":
                # Reverse the log transformation for UPV and SonReb
                prediction = np.exp(prediction)
            elif ndt_type == "rh":
                # Reverse the square root transformation for RH
                prediction = np.square(prediction)

            # Format the result to 2 decimal places
            if ndt_type == "rh":
                prediction_result = f"f<sub>c,cyl</sub> = {prediction:.2f} MPa"
            else:
                prediction_result = f"f<sub>c,cyl</sub> = {prediction:.2f} MPa"

            # Returning HTML formatted result
            return JsonResponse({'prediction': prediction_result})

            # return JsonResponse({'prediction': prediction})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'message': 'Send a POST request with input data.'})



# def predict_strength(features, ndt, scaler):
#     """Runs a prediction using all 10 trained models and returns the weighted average."""
    
#     # Define categorical feature indices (assumes known order)
#     categorical_feature_indices = [0, 1, 2, 3, 4, 5, 6]  # Adjust based on actual feature list
    
#     categorical_features = features[:, categorical_feature_indices]
#     numerical_features = np.delete(features, categorical_feature_indices, axis=1)

#     # Scale numerical features
#     scaled_numerical = scaler.transform(numerical_features)
    
#     # Combine scaled numerical + categorical into a DataFrame
#     features_combined = np.hstack((categorical_features, scaled_numerical))
#     features_df = pd.DataFrame(features_combined, columns=[f"f{i}" for i in range(features_combined.shape[1])])

#     # Run predictions using all models and average
#     predictions = [model.predict(features_df)[0] for model in models[ndt]]
#     avg_prediction = np.mean(predictions)
    
#     return float(avg_prediction)
