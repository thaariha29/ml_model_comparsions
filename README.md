# AI_ML_Task2_Model_Comparison
### Maincrafts Technology — AI & ML Internship | Task 2
**Feature Engineering, Model Optimization & Performance Comparison**

---

## Project Overview

This project builds an enhanced **House Price Prediction system** using the California Housing Dataset. It demonstrates a complete professional ML workflow — feature scaling, multi-model training, evaluation, and model selection.

---

## Files in This Repository

| File | Description |
|---|---|
| `AI_ML_Task2_Model_Comparison.ipynb` | Main Jupyter Notebook with all 9 steps |
| `california_housing.csv` | Dataset used for training and testing |
| `AI_ML_Task2_Report.pdf` | 1–2 page report: methodology, results, conclusions |
| `best_model.pkl` | Saved best model (Decision Tree) via joblib |
| `scaler.pkl` | Saved StandardScaler for future predictions |
| `README.md` | This file |

---

## Dataset

**California Housing Dataset** — 20,640 samples, 8 features + 1 target.

| Feature | Description |
|---|---|
| MedInc | Median income in block group |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Block group latitude |
| Longitude | Block group longitude |
| **HousePrice** | **TARGET — Median house value (×$100k)** |

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Upload `AI_ML_Task2_Model_Comparison.ipynb` and `california_housing.csv` to Colab
2. Run all cells top to bottom

### Option 2 — Local (Jupyter)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook AI_ML_Task2_Model_Comparison.ipynb
```

> **Note:** The notebook loads data from `california_housing.csv` by default.  
> If you want to fetch live from scikit-learn, change Step 2 to use `fetch_california_housing()`.

---

## ML Pipeline (8 Steps)

```
1. Import Libraries
2. Load Dataset
3. Separate Features & Target (X, y)
4. Feature Scaling — StandardScaler (mean=0, std=1)
5. Train–Test Split — 80% train / 20% test (random_state=42)
6. Train 3 Models — Linear Regression, Ridge, Decision Tree
7. Evaluate — RMSE & R² on test set
8. Visualise — Actual vs Predicted plots + bar charts
9. Save best model with joblib
```

---

## Results

| Model | RMSE ↓ | R² Score ↑ |
|---|---|---|
| Linear Regression | 0.4959 | 0.8393 |
| Ridge Regression | 0.4959 | 0.8393 |
| **Decision Tree** | **0.3265** | **0.9304** ✅ |

### Best Model: Decision Tree Regressor (max_depth=5)

**Justification:**
- Highest R² (0.9304) — explains 93% of variance in house prices
- Lowest RMSE (0.3265) — smallest average prediction error
- Captures **non-linear relationships** (income × location interactions) that linear models miss
- Ridge = Linear here because multicollinearity was not significant in this dataset

---

## Technologies Used

- Python 3.x
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- joblib
- Jupyter Notebook

---

## Author

**Thaariha**  
AI & ML Intern — Maincrafts Technology  
GitHub: [thaariha29](https://github.com/thaariha29)
