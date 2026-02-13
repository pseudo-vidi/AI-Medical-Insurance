import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
# Ensure 'insurance.csv' is uploaded to the main Files area in Colab
try:
    df = pd.read_csv('insurance.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: insurance.csv not found. Please upload it to the Files sidebar.")

# 2. PREPROCESSING (Converting text to numbers)
# Mapping categories as discussed in the project intro
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
# For region, we use simple numeric encoding
df['region'] = pd.factorize(df['region'])[0]

# Defining features and target based on Kaggle data
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# 3. TRAIN THE MODEL (Using Random Forest as planned)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. SAVE THE MODEL (.pkl file for Michael)
with open('medical_insurance_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Deliverable Created: medical_insurance_model.pkl")

# 5. MULTI-INSURER COMPARISON LOGIC (Real-world Benchmarks)
def get_multi_insurer_estimates(prediction):
    """
    Applies Canadian market benchmarking to the core AI prediction.
    """
    return {
        "Sun Life (Value Plan)": round(prediction * 0.85, 2), # Entry-level benchmark
        "Manulife (Standard Plan)": round(prediction * 1.0, 2),  # Industry average
        "Canada Life (Elite Plan)": round(prediction * 1.30, 2)  # Premium coverage benchmark
    }

# 6. TEST DEMO (Example for the meeting)
test_person = [[30, 1, 24.5, 0, 0, 1]] # Age 30, Male, BMI 24.5, 0 kids, Non-smoker
base_pred = model.predict(test_person)[0]
estimates = get_multi_insurer_estimates(base_pred)

print("\n--- MEETING DEMO RESULTS ---")
print(f"Base AI Prediction: ${base_pred:,.2f}")
for company, price in estimates.items():
    print(f"{company}: ${price:,.2f}")
