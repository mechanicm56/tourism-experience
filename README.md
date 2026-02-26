# 🌍 Tourism Experience Analytics System

## Overview
An end-to-end machine learning project that analyzes tourism data to:
- Predict attraction ratings
- Classify visit modes
- Recommend personalized attractions
- Visualize tourism trends

## 🚀 Live Demo (Streamlit App)

🔗 **Streamlit Demo:**  
👉 https://tourism-experience.streamlit.app/

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

- Create a virtual environment using virtualenv

- Install dependencies
```bash
pip install -r requirements.txt
```
- Generate required files and models using tourism_analysis.ipynb file

- At last, run the App
```bash
streamlit run app.py
```

## Future Enhancements
- Register models in MLflow Model Registry
- Load models directly from MLflow into Streamlit
- Deploy MLflow tracking to cloud (DagsHub)