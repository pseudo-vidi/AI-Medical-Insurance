import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. LOAD DATA
# We use pandas to read the raw CSV file containing insurance records.
try:
    df = pd.read_csv('insurance.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: insurance.csv not found.")

# 2. PREPROCESSING
# ML models require numerical input. We convert categorical text data into integers.
# Sex and Smoker are binary (0 or 1).
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# 'region' has 4 categories. factorize() assigns a unique number to each (0, 1, 2, 3).
df['region'] = pd.factorize(df['region'])[0]

# Split data into Features (X) and Target (y - the price we want to predict).
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Split the data: 80% for training the AI, 20% for testing its accuracy on new data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. INITIALIZE MODELS
# We define three different algorithms to see which one handles the data best.
models = {
    # Random Forest: A collection of decision trees that averages results.
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    
    # Linear Regression: Predicts a straight-line relationship between inputs and output.
    "LinearRegression": LinearRegression(),
    
    # Gradient Boosting: Builds trees one by one, each correcting the previous tree's errors.
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

results = {}

# 4. TRAINING AND EVALUATION LOOP
for name, model in models.items():
    # 'Fit' is the actual learning process where the model studies X_train to predict y_train.
    model.fit(X_train, y_train)
    
    # Test the model on the 20% of data it hasn't seen yet.
    predictions = model.predict(X_test)
    
    # Calculate R2 Score (Accuracy: 1.0 is perfect) and MAE (Average error in dollars).
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    results[name] = {"R2": round(r2, 4), "MAE": round(mae, 2)}
    
    # 5. EXPORT MODELS
    # Save each trained "brain" to a file so it can be reused in other scripts without retraining.
    filename = f'medical_insurance_{name.lower()}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")

# 6. DISPLAY PERFORMANCE
# Print a table to compare which model performed best.
print("\nModel Comparison Results:")
print(pd.DataFrame(results).T)
