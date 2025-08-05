
# movie_revenue_app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Movie Revenue Predictor", layout="centered")

# 🎨 Custom UI Styling with white text
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
        background-size: cover;
        background-attachment: fixed;
        color: white;
        font-weight: 400;
    }
    header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.5);
    }
    [data-testid="stSidebarNav"]::before {
        content: "🎬 MRP";
        margin-left: 10px;
        margin-top: 10px;
        font-size: 24px;
        position: relative;
        top: 10px;
    }
    .stMarkdown, .stTextInput, .stNumberInput label, .stButton button, .stSelectbox, .stRadio label, .stMultiSelect label {
        color: white !important;
    }
    </style>
    """ ,
    unsafe_allow_html=True
)

# 🖼️ Logo and Title
st.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Clapboard.svg", width=60)
st.title("🎬 Movie Revenue Predictor")
st.markdown("Upload movie data, select features, and predict revenue using ML models.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("boxoffice.csv")
    df.columns = [col.lower().strip() for col in df.columns]
    return df

df = load_data()

# Numeric columns for selection
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'world_revenue' in numeric_cols:
    numeric_cols.remove('world_revenue')

# Feature selection
st.subheader("🧮 Feature Selection")
selected_features = st.multiselect("Choose features for prediction:", numeric_cols, default=['budget', 'opening_revenue'])

# Model selection
model_type = st.radio("📌 Select Model", ["Linear Regression", "Random Forest"])

if selected_features:
    df = df.dropna(subset=['world_revenue'])

    X = df[selected_features]
    y = df['world_revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, selected_features)
    ])

    if model_type == "Linear Regression":
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
    else:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    model.fit(X_train, y_train)

    # 🎯 Manual Input for Prediction
    st.subheader("📝 Predict Revenue for a New Movie")
    user_input = {}

    for feature in selected_features:
        user_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, step=1000000.0)

    if st.button("Predict Revenue"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted World Revenue: ${prediction:,.2f}")

        # Download button
        result_df = input_df.copy()
        result_df["Predicted_World_Revenue"] = prediction
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Prediction", data=csv, file_name="predicted_revenue.csv", mime="text/csv")
else:
    st.warning("Select at least one feature to proceed.")
