import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# ==========================================
# 1. DATA LOADING
# ==========================================
# Load the insurance dataset containing health and cost information.
try:
    df = pd.read_csv('insurance.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: insurance.csv not found. Please ensure the dataset is in the same directory.")
    exit()

# ==========================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
# Convert categorical text data (strings) into numbers so the ML models can process them.
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# 'region' is converted into unique integers (0-3) using factorization.
df['region'] = pd.factorize(df['region'])[0]

# Features (X) are the independent variables; Target (y) is the premium cost (charges).
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Split data: 80% for training the models, 20% for testing their accuracy later.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FEATURE SCALING: 
# Neural networks (DNN/LSTM) are sensitive to the scale of input data. 
# StandardScaler ensures all features have a mean of 0 and a variance of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MULTI-INSURER PRICING LOGIC
# ==========================================
# This function applies different business multipliers to the raw AI prediction
# to simulate quotes from multiple insurance providers.
def get_multi_insurer_estimates(prediction):
    # Ensure prediction is a scalar (Keras models often return arrays)
    if isinstance(prediction, np.ndarray):
        prediction = prediction.item()
    return {
        "Sun Life (Value Plan)": round(prediction * 0.85, 2),   # Affiliate tier discount
        "Manulife (Standard Plan)": round(prediction * 1.0, 2), # Base AI prediction
        "Canada Life (Elite Plan)": round(prediction * 1.30, 2) # Comprehensive tier premium
    }

# ==========================================
# 4. DEEP LEARNING MODEL DEFINITIONS
# ==========================================
# Functions to build Deep Learning architectures using TensorFlow/Keras.

def build_dnn(input_dim):
    """Builds a Dense Neural Network with two hidden layers."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'), # High capacity layer to capture complex patterns
        Dense(32, activation='relu'), # Bottleneck layer for consolidation
        Dense(1)                      # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm(input_dim):
    """Builds an LSTM network. Note: LSTMs expect 3D sequence data."""
    model = Sequential([
        # LSTM expects input shape: (batch_size, time_steps, features)
        # Here we treat each row as a single time step sequence.
        Input(shape=(1, input_dim)),
        LSTM(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================
# 5. MODEL CONFIGURATION
# ==========================================
# A mix of Scikit-learn (traditional) and Keras (Deep Learning) model types.
models_config = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "DNN": "dnn",
    "LSTM": "lstm"
}

results = []

# ==========================================
# 6. TRAINING & EVALUATION LOOP
# ==========================================
print("\nStarting Training Pipeline for 5 Algorithms...")

for name, model_type in models_config.items():
    print(f"-> Training {name}...")
    
    # Handle Deep Learning Models (Keras)
    if name in ["DNN", "LSTM"]:
        if name == "DNN":
            model = build_dnn(X_train_scaled.shape[1])
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
            preds = model.predict(X_test_scaled).flatten()
        else: # LSTM
            model = build_lstm(X_train_scaled.shape[1])
            # Reshape data to 3D for LSTM: (Samples, Time_Steps, Features)
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
            preds = model.predict(X_test_lstm).flatten()
    
    # Handle Traditional Machine Learning Models (Sklearn)
    else:
        model = model_type
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    # Performance Evaluation
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    results.append({"Model": name, "R2": r2, "MAE": mae})

    # ==========================================
    # 7. BUNDLING & SERIALIZATION
    # ==========================================
    # Each bundle (.pkl) contains:
    # 1. The trained model
    # 2. The scaler (must be used to scale input data during inference)
    # 3. The pricing logic function
    # 4. Performance metadata
    model_bundle = {
        "model_object": model,
        "scaler_object": scaler,
        "pricing_logic": get_multi_insurer_estimates,
        "algorithm_name": name,
        "metrics": {"R2": r2, "MAE": mae}
    }
    
    # Save the bundle as a binary file
    filename = f'medical_insurance_{name.lower()}_bundle.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"   Done! Created bundle: {filename} (R2 Score: {r2:.4f})")

# ==========================================
# 8. FINAL PERFORMANCE SUMMARY
# ==========================================
print("\n" + "="*40)
print("FINAL PERFORMANCE LEADERBOARD")
print("="*40)
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print(results_df.to_string(index=False))

# Export raw stats for further analysis if needed
with open('model_stats.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nDeploy Ready: All 5 models are bundled with unified pricing logic and feature scalers.")
