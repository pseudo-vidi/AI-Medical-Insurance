# Medical Insurance Premium Predictor: Model Comparison Report

This report evaluates the performance of five different regression algorithms used to predict medical insurance premiums based on health and lifestyle factors.

## Performance Metrics Summary

The models are evaluated using the **R² Score** (coefficient of determination) and **Mean Absolute Error (MAE)**.

| Rank | Model | R² Score | MAE ($) |
| :--- | :--- | :--- | :--- |
| 1 | **Gradient Boosting** | 0.8781 | 2,446.93 |
| 2 | **Random Forest** | 0.8643 | 2,513.90 |
| 3 | **LSTM (Deep Learning)** | 0.8091 | 3,614.38 |
| 4 | **DNN (Deep Learning)** | 0.7909 | 4,015.39 |
| 5 | **Linear Regression** | 0.7833 | 4,186.51 |

---

## Model Analysis & Ranking

### 1. Gradient Boosting (Winner) 🏆
*   **Why it's better**: Gradient Boosting sequentially corrects errors from previous trees, making it extremely effective for tabular data like this insurance dataset. It achieved the highest accuracy and the lowest error margin.
*   **Best For**: Production deployment where accuracy is the primary goal.

### 2. Random Forest
*   **Analysis**: A very strong second. Being an ensemble of many decision trees, it handles non-linear relationships well and is less prone to overfitting than a single decision tree.
*   **Best For**: Robust predictions across diverse health profiles.

### 3. LSTM (Long Short-Term Memory)
*   **Analysis**: Surprisingly effective for tabular data in this context. While LSTMs are usually for time-series data, they can capture complex feature interactions. However, they are "overkill" for a simple dataset of 1,338 rows.
*   **Best For**: Experimental purposes or if sequential/longitudinal health data were available.

### 4. DNN (Dense Neural Network)
*   **Analysis**: Performed reasonably well but was outperformed by tree-based methods. Neural networks typically require much larger datasets (10k+ rows) to truly shine.
*   **Best For**: Scalability if the dataset grows significantly.

### 5. Linear Regression
*   **Analysis**: The baseline model. While transparent and fast, it assumes linear relationships that don't always exist in complex health data (like the impact of smoking combined with high BMI).
*   **Best For**: Understanding general trends and logic.

---

## Conclusion
For this specific medical insurance dataset, **Gradient Boosting** is the superior choice. The deep learning models (LSTM/DNN) are promising but require more data to surpass the efficiency of Gradient Boosting and Random Forest on this tabular scale.
