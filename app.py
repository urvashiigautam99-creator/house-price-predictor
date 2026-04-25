import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction")
st.markdown("Predict California house prices using Machine Learning")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.header("Enter House Details")
MedInc     = st.sidebar.slider("Median Income (in $10k)", 0.5, 15.0, 5.0)
HouseAge   = st.sidebar.slider("House Age (years)", 1, 52, 20)
AveRooms   = st.sidebar.slider("Average Rooms", 1.0, 15.0, 5.0)
AveBedrms  = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Block Population", 100, 5000, 1000)
AveOccup   = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
Latitude   = st.sidebar.slider("Latitude", 32.0, 42.0, 36.0)
Longitude  = st.sidebar.slider("Longitude", -124.0, -114.0, -120.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                         Population, AveOccup, Latitude, Longitude]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted House Value", value=f"${prediction * 100000:,.0f}")
    st.success("Prediction complete!")

with col2:
    st.subheader("Your Input Summary")
    input_df = pd.DataFrame({
        "Feature": ["Median Income", "House Age", "Avg Rooms", "Avg Bedrooms",
                    "Population", "Avg Occupancy", "Latitude", "Longitude"],
        "Value": [MedInc, HouseAge, AveRooms, AveBedrms,
                  Population, AveOccup, Latitude, Longitude]
    })
    st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.subheader("📊 Dataset Overview")
data = fetch_california_housing(as_frame=True)
df = data.frame

col3, col4 = st.columns(2)

with col3:
    st.write("**Price Distribution**")
    fig, ax = plt.subplots()
    ax.hist(df["MedHouseVal"], bins=50, color="#7F77DD", edgecolor="white")
    ax.set_xlabel("House Value ($100k)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col4:
    st.write("**Income vs Price**")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.1, color="#1D9E75", s=5)
    ax2.set_xlabel("Median Income")
    ax2.set_ylabel("House Value")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit | California Housing Dataset")