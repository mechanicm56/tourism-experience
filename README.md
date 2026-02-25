# 🌍 Tourism Experience Analytics System

## Overview
An end-to-end machine learning project that analyzes tourism data to:
- Predict attraction ratings
- Classify visit modes
- Recommend personalized attractions
- Visualize tourism trends

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- SQL (data preparation)
- Matplotlib, Seaborn

## Features
- Regression: Predict user ratings
- Classification: Predict visit mode
- Recommendation System (Collaborative + Profile-based)
- Interactive Streamlit dashboard
- Business-focused insights

## How to Run
```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_regression.py
python src/train_classification.py
streamlit run app.py
```

## Future Enhancements
- Register models in MLflow Model Registry
- Load models directly from MLflow into Streamlit
- Add versioning & staging (Production / Staging)
- Deploy MLflow tracking to cloud (DagsHub)