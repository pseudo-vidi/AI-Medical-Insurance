import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. DATA LOADING
# Load the insurance dataset containing health and cost information.
try:
    df = pd.read_csv('insurance.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: insurance.csv not found.")

# 2. PREPROCESSING
# Convert categorical text data (strings) into numbers so the ML models can process them.
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
# 'region' is converted into unique integers (0-3).
df['region'] = pd.factorize(df['region'])[0]

# Features (X) are the independent variables; Target (y) is the premium cost.
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Split data: 80% for training the 'brain', 20% for testing its accuracy later.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DEFINE MULTI-INSURER LOGIC
# This function calculates tiered pricing based on a single AI prediction.
# We define it here so it can be packaged inside each .pkl file.
def get_multi_insurer_estimates(prediction):
    return {
        "Sun Life (Value Plan)": round(prediction * 0.85, 2), # 15% discount benchmark
        "Manulife (Standard Plan)": round(prediction * 1.0, 2),  # Base AI prediction
        "Canada Life (Elite Plan)": round(prediction * 1.30, 2)  # 30% premium benchmark
    }

# 4. INITIALIZE THE THREE REGRESSION MODELS
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# 5. TRAINING AND BUNDLING PROCESS
for name, model in models.items():
    # Model learns the relationship between health factors and insurance charges.
    model.fit(X_train, y_train)
    
    # BUNDLE CREATION:
    # Instead of just saving the model, we create a dictionary.
    # This stores the model AND the function we defined in Step 3.
    model_bundle = {
        "model_object": model,          # The trained regression model
        "pricing_logic": get_multi_insurer_estimates, # The comparison function
        "algorithm_name": name          # Metadata about which algorithm was used
    }
    
    # SAVE THE BUNDLE
    # We use 'wb' (write binary) to serialize the entire dictionary into a .pkl file.
    filename = f'medical_insurance_{name.lower()}_bundle.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"Successfully created and bundled: {filename}")

print("\nDeployment ready: Each .pkl now contains both the AI model and the pricing logic.")
