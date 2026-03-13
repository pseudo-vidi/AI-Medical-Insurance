# Medical Insurance AI Premium Predictor (Multi-Model Edition)

## Project Overview
This AI/ML initiative estimates medical insurance premiums based on individual health and lifestyle factors. Designed as a proof-of-concept for the Canadian market, the system provides a comparative analysis across multiple insurance providers using five distinct regression methods (including Deep Learning) to ensure the most accurate cost estimation.

## Key Deliverables
* **Serialized Model Bundles:** Five trained models (`RandomForest`, `LinearRegression`, `GradientBoosting`, `DNN`, and `LSTM`) saved as `.pkl` bundles.
* **Performance Intelligence:** Integrated `model_stats.pkl` for raw performance logging and `model_comparison.md` for human-readable rankings.
* **Integrated Logic:** Each bundle contains the predictive "brain," a feature scaler, and the multi-insurer pricing logic for seamless deployment.
* **Multi-Insurer Framework:** A system providing simultaneous quotes for Sun Life, Manulife, and Canada Life.

## Technology Stack
The project leverages the following industry-standard AI/ML libraries:
* **TensorFlow:** The "Brain Builder" for Deep Learning, used for constructing DNN and LSTM architectures.
* **Scikit-learn:** Performs feature scaling, data splitting, and handles traditional regression models (Random Forest, Linear Regression, Gradient Boosting).
* **Pandas:** Manages high-performance data structures for loading and preprocessing the insurance dataset.

## Data Strategy
The model utilizes ethically sourced data to maintain academic and professional integrity:
* **Primary Engine:** Kaggle Medical Insurance Dataset (Age, Sex, BMI, Children, Smoker Status, Region).
* **Preprocessing:** Categorical data is mapped (Sex, Smoker) and factorized (Region). Features are standardized using `StandardScaler` to ensure compatibility with deep learning architectures.

## Multi-Insurer Logic
The AI model acts as a "Base Predictor." Each bundle includes a pricing function that adjusts this base rate using real-world Canadian insurer benchmarks:
* **Sun Life (Value Plan):** Optimized for entry-level affordability (85% of base).
* **Manulife (Standard Plan):** Focused on industry-average pricing (100% of base).
* **Canada Life (Elite Plan):** Calculated for comprehensive coverage (130% of base).

## Model Comparison & Performance
The system evaluates three different approaches to find the best fit for insurance data:
1.  **Linear Regression:** Provides a transparent, baseline mathematical relationship.
2.  **Random Forest:** An ensemble method that handles non-linear health risks effectively.
3.  **Gradient Boosting:** A sequential learning method providing the highest accuracy for this tabular data.
4.  **Deep Neural Network (DNN):** A multi-layer architecture for capturing complex, non-linear relationships at scale.
5.  **LSTM (Long Short-Term Memory):** A recurrent architecture typically used for sequences, applied here to capture deep feature dependencies.



## Regulatory & HIPAA Compliance
| Process | Implementation |
| :--- | :--- |
| **Data De-identification** | Uses anonymized open-source data; no Protected Health Information (PHI) is stored. |
| **Safe Harbor Method** | Removal of all 18 HIPAA identifiers from training sets to ensure privacy by design. |
| **Data Minimization** | Processes only the "Minimum Necessary" fields required for an accurate estimate. |

## How to Use

### 1. Training
Run `train_model.py` to process `insurance.csv`. This will generate five bundles:
* `medical_insurance_randomforest_bundle.pkl`
* `medical_insurance_linearregression_bundle.pkl`
* `medical_insurance_gradientboosting_bundle.pkl`
* `medical_insurance_dnn_bundle.pkl`
* `medical_insurance_lstm_bundle.pkl`

Additionally, it produces `model_stats.pkl` (serialized metrics) and `model_comparison.md` (leaderboard).

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
