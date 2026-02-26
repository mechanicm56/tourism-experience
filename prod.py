import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import gdown

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# GOOGLE DRIVE FILE IDS
# --------------------------------------------------
FILE_IDS = {
    "clf_rf": "1rKp0i2zJ7FiH4_yi18fAJd4jTc0scXSI",
    "reg_rf": "17SSG57s0guWLJm46cQwBWxrfyJRitTUe"
}

# --------------------------------------------------
# HELPER: Download file from Drive if not exists
# --------------------------------------------------
def download_from_drive(file_id, output_path):
    if not output_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
    return output_path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Tourism Intelligence Platform",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 Tourism Intelligence Platform")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data/processed/tourism_master.csv")

df = load_data()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    clf_rf_path = download_from_drive(FILE_IDS["clf_rf"], BASE_DIR / "models/clf_RandomForest.pkl")
    reg_rf_path = download_from_drive(FILE_IDS["reg_rf"], BASE_DIR / "scaler.pkl")

    clf_models = {
        "XGBoost": joblib.load(BASE_DIR / "models/clf_XGBoost.pkl"),
        "LightGBM": joblib.load(BASE_DIR / "models/clf_LightGBM.pkl"),
        "RandomForest": joblib.load(clf_rf_path),
        "LogisticRegression": joblib.load(BASE_DIR / "models/clf_LogisticRegression.pkl")
    }

    reg_models = {
        "XGBoost": joblib.load(BASE_DIR / "models/reg_XGBoost.pkl"),
        "LightGBM": joblib.load(BASE_DIR / "models/reg_LightGBM.pkl"),
        "RandomForest": joblib.load(reg_rf_path)
    }

    clf_encoders = joblib.load(BASE_DIR / "models/clf_encoders.pkl")
    reg_encoders = joblib.load(BASE_DIR / "models/reg_encoders.pkl")

    clf_features = joblib.load(BASE_DIR / "models/clf_features.pkl")
    reg_features = joblib.load(BASE_DIR / "models/reg_features.pkl")

    return clf_models, reg_models, clf_encoders, reg_encoders, clf_features, reg_features

clf_models, reg_models, clf_encoders, reg_encoders, clf_features, reg_features = load_models()

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Home",
        "Model Comparison",
        "VisitMode Prediction",
        "Rating Prediction",
        "Recommendations"
    ]
)

# =================================================
# HOME
# =================================================
if page == "Home":

    st.header("Welcome to the Travel Intelligence Dashboard")
    st.write("""
    This system predicts:
    - Visit Mode (Business, Family, Couples, Friends, etc.)
    - Attraction Rating
    """)

# =================================================
# MODEL COMPARISON
# =================================================
elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    reg_comp = pd.read_csv(BASE_DIR / "models/regression_model_comparison.csv")
    clf_comp = pd.read_csv(BASE_DIR / "models/classification_model_comparison.csv")

    st.subheader("Regression Models (Rating Prediction)")
    st.dataframe(reg_comp)

    fig_reg = px.bar(reg_comp, x="Model", y="RMSE")
    st.plotly_chart(fig_reg, width='stretch')

    st.subheader("Classification Models (VisitMode Prediction)")
    st.dataframe(clf_comp)

    fig_clf = px.bar(clf_comp, x="Model", y="Accuracy")
    st.plotly_chart(fig_clf, width='stretch')

# =================================================
# VISIT MODE PREDICTION
# =================================================
elif page == "VisitMode Prediction":

    st.header("Predict Visit Mode")

    # -----------------------------
    # Dynamic Country → Region → City
    # -----------------------------

    # country = st.selectbox("Country", sorted(df["Country"].unique()))

    # st.write("Model features:", clf_features)
    # st.write("Input features:", df.columns.tolist())

    country_list = (
        df["Country"]
        .dropna()
        .astype(str)
        .unique()
    )

    country = st.selectbox("Country", sorted(country_list))

    region_options = (
        df[df["Country"] == country]["Region"]
        .dropna()
        .astype(str)
        .unique()
    )

    region = st.selectbox("Region", sorted(region_options))


    city_options = (
        df[(df["Country"] == country) &
        (df["Region"] == region)]["CityName"]
        .dropna()
        .astype(str)
        .unique()
    )

    city = st.selectbox("City", sorted(city_options))

    # region_options = df[df["Country"] == country]["Region"].unique()
    # region = st.selectbox("Region", sorted(region_options))

    # city_options = df[
    #     (df["Country"] == country) &
    #     (df["Region"] == region)
    # ]["CityName"].unique()

    # city = st.selectbox("City", sorted(city_options))

    st.divider()

    # Attraction info
    attraction_type = st.selectbox("Attraction Type", df["AttractionType"].unique())
    attraction_country = st.selectbox("Attraction Country", df["AttractionCountry"].unique())
    attraction_city = st.selectbox("Attraction City", df["AttractionCity"].unique())

    attraction_popularity = st.slider("Attraction Popularity", 1, 10000, 100)
    attraction_avg_rating = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0)

    visit_year = st.slider("Visit Year", 2015, 2026, 2024)
    visit_month = st.slider("Visit Month", 1, 12, 6)

    model_choice = st.selectbox("Select Model", list(clf_models.keys()))

    if st.button("Predict Visit Mode"):

        try:
            with st.spinner("Analyzing travel patterns..."):

                input_dict = {
                    "Country": country,
                    "Region": region,
                    "CityName": city,
                    "AttractionType": attraction_type,
                    "VisitYear": visit_year,
                    "VisitMonth": visit_month,
                }

                input_df = pd.DataFrame([input_dict])

                # Feature engineering
                input_df["Season"] = input_df["VisitMonth"].map({
                    12:"Winter",1:"Winter",2:"Winter",
                    3:"Spring",4:"Spring",5:"Spring",
                    6:"Summer",7:"Summer",8:"Summer",
                    9:"Autumn",10:"Autumn",11:"Autumn"
                })

                input_df["IsDomesticTrip"] = 0

                # Encode safely
                for col in input_df.columns:
                    if col in clf_encoders:
                        try:
                            input_df[col] = clf_encoders[col].transform(input_df[col])
                        except:
                            input_df[col] = 0

                # Ensure all required columns exist
                for col in clf_features:
                    if col not in input_df.columns:
                        input_df[col] = 0

                input_df = input_df[clf_features]

                model = clf_models["RandomForest"]  # change if needed

                prediction = model.predict(input_df)[0]

                st.success(f"Predicted Visit Mode: {prediction}")

                # Probabilities
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    classes = model.classes_

                    proba_df = pd.DataFrame({
                        "VisitMode": classes,
                        "Probability": probs
                    }).sort_values("Probability", ascending=False).head(3)

                    fig = px.bar(
                        proba_df,
                        x="Probability",
                        y="VisitMode",
                        orientation="h"
                    )

                    st.plotly_chart(fig, width='stretch')

        except Exception as e:
            # print(e)
            st.error(f"Prediction Error: {e}")

# =================================================
# RATING PREDICTION
# =================================================
elif page == "Rating Prediction":

    st.header("Predict Attraction Rating")

    attraction_type = st.selectbox("Attraction Type", df["AttractionType"].unique())
    attraction_country = st.selectbox("Attraction Country", df["AttractionCountry"].unique())
    attraction_city = st.selectbox("Attraction City", df["AttractionCity"].unique())

    visit_year = st.slider("Visit Year", 2015, 2026, 2024)
    visit_month = st.slider("Visit Month", 1, 12, 6)

    model_choice = st.selectbox("Select Regression Model", list(reg_models.keys()))

    if st.button("Predict Rating"):

        try:
            with st.spinner("Estimating rating..."):

                input_dict = {
                    "AttractionType": attraction_type,
                    "VisitYear": visit_year,
                    "VisitMonth": visit_month,
                }

                input_df = pd.DataFrame([input_dict])

                input_df["Season"] = input_df["VisitMonth"].map({
                    12:"Winter",1:"Winter",2:"Winter",
                    3:"Spring",4:"Spring",5:"Spring",
                    6:"Summer",7:"Summer",8:"Summer",
                    9:"Autumn",10:"Autumn",11:"Autumn"
                })

                # Encode safely
                for col in input_df.columns:
                    if col in reg_encoders:
                        try:
                            input_df[col] = reg_encoders[col].transform(input_df[col])
                        except:
                            input_df[col] = 0

                # Ensure required columns exist
                for col in reg_features:
                    if col not in input_df.columns:
                        input_df[col] = 0

                input_df = input_df[reg_features]

                model = reg_models["RandomForest"]

                prediction = model.predict(input_df)[0]

                st.success(f"Predicted Rating: {round(float(prediction),2)} / 5")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =================================================
# RECOMMENDATION SYSTEM
# =================================================
elif page == "Recommendations":

    st.header("Personalized Attraction Recommendations")

    user_id = st.selectbox("Select User", df["UserId"].unique())

    if st.button("Generate Recommendations"):

        with st.spinner("Finding similar travelers and ranking attractions..."):

            # User history
            user_history = df[df["UserId"] == user_id][
                ["Attraction", "Rating", "VisitMode"]
            ]

            st.subheader("User Visit History")
            st.dataframe(user_history)

            # User-item matrix
            user_item = df.pivot_table(
                index="UserId",
                columns="Attraction",
                values="Rating"
            ).fillna(0)

            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(user_item)
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=user_item.index,
                columns=user_item.index
            )

            similar_users = (
                similarity_df[user_id]
                .sort_values(ascending=False)
                .iloc[1:6]
                .index
            )

            # Collaborative filtering
            similar_user_ratings = user_item.loc[similar_users]

            collaborative_scores = (
                similar_user_ratings.mean()
                .sort_values(ascending=False)
            )

            visited_attractions = user_history["Attraction"].values
            collaborative_scores = collaborative_scores.drop(
                visited_attractions,
                errors="ignore"
            )

            # Popularity boost
            popularity = df.groupby("Attraction")["UserId"].count()
            popularity_norm = popularity / popularity.max()

            hybrid_score = (
                0.7 * collaborative_scores +
                0.3 * popularity_norm
            ).dropna().sort_values(ascending=False)

            top_recommendations = hybrid_score.head(10)

            rec_df = pd.DataFrame({
                "Attraction": top_recommendations.index,
                "Score": top_recommendations.values
            })

            st.subheader("Recommended Attractions")
            st.dataframe(rec_df)

            fig = px.bar(
                rec_df,
                x="Score",
                y="Attraction",
                orientation="h"
            )

            st.plotly_chart(fig, width='stretch')