import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parent

print(BASE_DIR)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Travel Intelligence Platform",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 Travel Intelligence & Recommendation System")

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
    reg_models = {
        "XGBoost": joblib.load(BASE_DIR / "models/reg_XGBoost.pkl"),
        "LightGBM": joblib.load(BASE_DIR / "models/reg_LightGBM.pkl"),
        "RandomForest": joblib.load(BASE_DIR / "models/reg_RandomForest.pkl")
    }

    clf_models = {
        "XGBoost": joblib.load(BASE_DIR / "models/clf_XGBoost.pkl"),
        "LightGBM": joblib.load(BASE_DIR / "models/clf_LightGBM.pkl"),
        "RandomForest": joblib.load(BASE_DIR / "models/clf_RandomForest.pkl"),
        "LogisticRegression": joblib.load(BASE_DIR / "models/clf_LogisticRegression.pkl")
    }

    reg_encoders = joblib.load(BASE_DIR / "models/reg_encoders.pkl")
    clf_encoders = joblib.load(BASE_DIR / "models/clf_encoders.pkl")

    reg_features = joblib.load(BASE_DIR / "models/reg_features.pkl")
    clf_features = joblib.load(BASE_DIR / "models/clf_features.pkl")

    return reg_models, clf_models, reg_encoders, clf_encoders, reg_features, clf_features

reg_models, clf_models, reg_encoders, clf_encoders, reg_features, clf_features = load_models()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filters")

country_filter = st.sidebar.multiselect(
    "Select Attraction Country",
    df["AttractionCountry"].unique(),
    default=df["AttractionCountry"].unique()
)

type_filter = st.sidebar.multiselect(
    "Select Attraction Type",
    df["AttractionType"].unique(),
    default=df["AttractionType"].unique()
)

filtered_df = df[
    (df["AttractionCountry"].isin(country_filter)) &
    (df["AttractionType"].isin(type_filter))
]

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analytics",
    "🌍 Geo Map",
    "🤖 Prediction",
    "🎯 Recommendation"
])

# =================================================
# TAB 1 — ANALYTICS
# =================================================
with tab1:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Users", filtered_df["UserId"].nunique())
    col2.metric("Total Attractions", filtered_df["Attraction"].nunique())
    col3.metric("Avg Rating", round(filtered_df["Rating"].mean(),2))
    col4.metric("Countries", filtered_df["AttractionCountry"].nunique())

    st.subheader("Top Attractions")

    top_attr = (
        filtered_df.groupby("Attraction")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(top_attr, x="Rating", y="Attraction", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Comparison")

    reg_comp = pd.read_csv(BASE_DIR / "models/regression_model_comparison.csv")
    clf_comp = pd.read_csv(BASE_DIR / "models/classification_model_comparison.csv")

    st.write("Regression Models")
    st.dataframe(reg_comp)

    st.write("Classification Models")
    st.dataframe(clf_comp)

# =================================================
# TAB 2 — GEO MAP
# =================================================
with tab2:

    geo_df = (
        filtered_df.groupby("AttractionCountry")
        .agg(AvgRating=("Rating","mean"),
             VisitCount=("Attraction","count"))
        .reset_index()
    )

    fig = px.choropleth(
        geo_df,
        locations="AttractionCountry",
        locationmode="country names",
        color="VisitCount",
        hover_data=["AvgRating"],
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 3 — PREDICTION
# =================================================
with tab3:

    prediction_type = st.selectbox(
        "Choose Prediction Type",
        ["Rating (Regression)", "VisitMode (Classification)"]
    )

    model_choice = st.selectbox(
        "Select Model",
        list(reg_models.keys()) if "Rating" in prediction_type else list(clf_models.keys())
    )

    st.subheader("Input Features")

    input_data = {}

    for col in reg_features if "Rating" in prediction_type else clf_features:
        if col in df.columns:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        if "Rating" in prediction_type:
            for col in input_df.columns:
                if col in reg_encoders:
                    input_df[col] = reg_encoders[col].transform(input_df[col])
            input_df = input_df[reg_features]
            prediction = reg_models[model_choice].predict(input_df)[0]
            st.success(f"Predicted Rating: {round(prediction,2)}")

        else:
            for col in input_df.columns:
                if col in clf_encoders:
                    input_df[col] = clf_encoders[col].transform(input_df[col])
            input_df = input_df[clf_features]
            prediction = clf_models[model_choice].predict(input_df)[0]
            st.success(f"Predicted Visit Mode: {prediction}")

# =================================================
# TAB 4 — HYBRID RECOMMENDATION
# =================================================
with tab4:

    user_id = st.selectbox("Select User", df["UserId"].unique())

    user_item = df.pivot_table(
        index="UserId",
        columns="Attraction",
        values="Rating"
    ).fillna(0)

    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(user_item)
    sim_df = pd.DataFrame(
        similarity,
        index=user_item.index,
        columns=user_item.index
    )

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:6].index
    recommendations = (
        user_item.loc[similar_users]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    st.write("Recommended Attractions:")
    for rec in recommendations:
        st.success(rec)

# -------------------------------------------------
# DOWNLOAD DATA
# -------------------------------------------------
st.sidebar.download_button(
    "⬇ Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_data.csv"
)