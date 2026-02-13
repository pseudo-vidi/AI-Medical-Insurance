# Medical Insurance AI Premium Predictor

## Project Overview

This project is an AI/ML initiative designed to estimate medical insurance premiums based on individual health and lifestyle factors. Developed as a proof-of-concept for an external partnership, the system provides a comparative analysis across multiple insurance providers to help users understand cost variations in the Canadian market.

## Key Deliverables

* **Validated ML Model:** A trained `RandomForestRegressor` saved as a serialized `.pkl` file for production use.
* **Source Code Repository:** Complete scripts for data cleaning, feature engineering, and model training.
* **Multi-Insurer Framework:** A logic-based system providing simultaneous quotes for at least three major insurers.

## Data Strategy

The model is built on ethically sourced, public data to maintain academic and professional integrity:

* **Primary Engine:** Kaggle Medical Insurance Dataset (Age, Sex, BMI, Children, Smoker Status, Region).
* **Contextual Support:** Canadian Open Government Data used to justify regional risk and cost variations.

## Multi-Insurer Logic

As requested, the system integrates a value proposition of sharing multiple estimates. The AI model acts as a "Base Predictor," which is then adjusted using real-world Canadian insurer benchmarks:

* **Sun Life (Value Tier):** Optimized for entry-level affordability.
* **Manulife (Standard Tier):** Focused on industry-average wellness-based pricing.
* **Canada Life (Premium Tier):** Calculated for high-limit comprehensive coverage.

## Regulatory & HIPAA Compliance

To ensure the software meets government regulatory standards, the following processes are implemented:

| Process | Implementation |
| --- | --- |
| **Data De-identification** | Strictly using anonymized open-source data; no Protected Health Information (PHI) is collected or stored.
| **Safe Harbor Method** | Removal of all 18 HIPAA identifiers from training sets to ensure privacy by design. |
| **Data Minimization** | The model only processes the "Minimum Necessary" fields required for an accurate estimate.
| **Encryption Roadmap** | Future development includes AES-256 bit encryption at rest and TLS 1.2+ for data in transit. |

## How to Use

1. **Environment:** Run the provided scripts in a Python environment (VS Code) or Google Colab.
2. **Training:** Run `train_model.py` to process `insurance.csv` and generate the `medical_insurance_model.pkl` file.
3. **Inference:** Use the model to predict a base rate, which the system then converts into a multi-insurer comparison.

