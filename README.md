# Medical Insurance AI Premium Predictor (Multi-Model Edition)

## Project Overview
This AI/ML initiative estimates medical insurance premiums based on individual health and lifestyle factors. Designed as a proof-of-concept for the Canadian market, the system provides a comparative analysis across multiple insurance providers using three distinct regression methods to ensure the most accurate cost estimation.

## Key Deliverables
* **Serialized Model Bundles:** Three trained models (`RandomForest`, `LinearRegression`, and `GradientBoosting`) saved as `.pkl` bundles.
* **Integrated Logic:** Each bundle contains both the predictive "brain" and the multi-insurer pricing logic for seamless deployment.
* **Multi-Insurer Framework:** A system providing simultaneous quotes for Sun Life, Manulife, and Canada Life.

## Data Strategy
The model utilizes ethically sourced data to maintain academic and professional integrity:
* **Primary Engine:** Kaggle Medical Insurance Dataset (Age, Sex, BMI, Children, Smoker Status, Region).
* **Preprocessing:** Categorical data is mapped (Sex, Smoker) and factorized (Region) to ensure compatibility with all three regression algorithms.

## Multi-Insurer Logic
The AI model acts as a "Base Predictor." Each bundle includes a pricing function that adjusts this base rate using real-world Canadian insurer benchmarks:
* **Sun Life (Value Plan):** Optimized for entry-level affordability (85% of base).
* **Manulife (Standard Plan):** Focused on industry-average pricing (100% of base).
* **Canada Life (Elite Plan):** Calculated for comprehensive coverage (130% of base).

## Model Comparison & Performance
The system evaluates three different approaches to find the best fit for insurance data:
1.  **Linear Regression:** Provides a transparent, baseline mathematical relationship.
2.  **Random Forest:** An ensemble method that handles non-linear health risks effectively.
3.  **Gradient Boosting:** A sequential learning method often providing the highest accuracy for tabular data.



## Regulatory & HIPAA Compliance
| Process | Implementation |
| :--- | :--- |
| **Data De-identification** | Uses anonymized open-source data; no Protected Health Information (PHI) is stored. |
| **Safe Harbor Method** | Removal of all 18 HIPAA identifiers from training sets to ensure privacy by design. |
| **Data Minimization** | Processes only the "Minimum Necessary" fields required for an accurate estimate. |

## How to Use

### 1. Training
Run `train_model.py` to process `insurance.csv`. This will generate three files:
* `medical_insurance_randomforest_bundle.pkl`
* `medical_insurance_linearregression_bundle.pkl`
* `medical_insurance_gradientboosting_bundle.pkl`

### 2. Inference (Loading & Using)
To use any of the models in your application, use the following code structure:

```python
import pickle

# Load the desired bundle (e.g., Gradient Boosting)
with open('medical_insurance_gradientboosting_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

# Access the components from the bundle
my_model = bundle["model_object"]
calc_logic = bundle["pricing_logic"]

# 1. Get base prediction (ensure input data is preprocessed)
# Example: base_price = my_model.predict([[19, 0, 27.9, 0, 1, 3]])[0]
base_price = my_model.predict(user_input_data)[0]

# 2. Get insurer comparisons automatically
final_quotes = calc_logic(base_price)

print(f"Algorithm Used: {bundle['algorithm_name']}")
print(f"Comparison Results: {final_quotes}")
