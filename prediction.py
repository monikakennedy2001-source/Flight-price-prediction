import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

st.set_page_config(page_title="Flight Price App", layout="wide")

# --------------------
# Load Pickle Artifacts
# --------------------
@st.cache_data
def load_artifacts():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder_total_stops.pkl", "rb") as f:
        le_total_stops = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features_list = pickle.load(f)
    return model, le_total_stops, features_list

model, le_total_stops, features_list = load_artifacts()

# --------------------
# Load Dataset for EDA
# --------------------
@st.cache_data
def load_data():
    return pd.read_csv("Flight_Price.csv")

df_raw = load_data()

# --------------------
# EDA Page
# --------------------
def eda_page():
    st.title("🧠 Exploratory Data Analysis")

    st.subheader("1️⃣ Sample Data")
    st.write(df_raw.head())

    st.subheader("2️⃣ Null Values")
    st.write(df_raw.isnull().sum())

    st.subheader("3️⃣ Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(df_raw['Price'], bins=50, kde=True, ax=ax1, color='teal')
    ax1.set_title("Distribution of Flight Prices")
    st.pyplot(fig1)

    st.subheader("4️⃣ Price by Airline")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.boxplot(x='Airline', y='Price', data=df_raw, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_title("Flight Price by Airline")
    st.pyplot(fig2)

    st.subheader("5️⃣ Price by Total Stops")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='Total_Stops', y='Price', data=df_raw, ax=ax3)
    ax3.set_title("Flight Price vs Number of Stops")
    st.pyplot(fig3)

    st.subheader("6️⃣ Correlation Matrix")
    df = df_raw.copy()
    df.dropna(inplace=True)
    df["Duration_mins"] = df["Duration"].apply(convert_duration_to_minutes)
    df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
    df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
    numeric_df = df[["Price", "Duration_mins", "Journey_day", "Journey_month"]]
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation Matrix")
    st.pyplot(fig4)

# --------------------
# Helper for duration conversion
# --------------------
def convert_duration_to_minutes(x):
    x = x.strip()
    if 'h' in x and 'm' in x:
        h, m = x.split('h')
        return int(h.strip()) * 60 + int(m.replace('m', '').strip())
    elif 'h' in x:
        return int(x.replace('h', '').strip()) * 60
    elif 'm' in x:
        return int(x.replace('m', '').strip())
    return 0

# --------------------
# Prediction Page
# --------------------
def prediction_page():
    st.title("✈️ Flight Price Prediction")

    # User inputs
    total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops'])
    journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=1)
    journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=1)
    dep_hour = st.slider("Departure Hour", 0, 23, 12)
    dep_minute = st.slider("Departure Minute", 0, 59, 0)
    arrival_hour = st.slider("Arrival Hour", 0, 23, 12)
    arrival_minute = st.slider("Arrival Minute", 0, 59, 0)
    duration_mins = st.number_input("Duration (minutes)", min_value=0, value=60)

    airline = st.selectbox("Airline", ['Jet Airways', 'IndiGo', 'Air India', 'SpiceJet', 'GoAir', 'Jet Airways Business'])
    add_info = st.selectbox("Additional Info", ['No info', 'In-flight meal not included', 'Meal provided'])

    # Encode Total Stops
    encoded_stops = le_total_stops.transform([total_stops])[0]

    # Create input feature vector
    input_dict = {
        'Journey_day': journey_day,
        'Journey_month': journey_month,
        'Dep_hour': dep_hour,
        'Dep_minute': dep_minute,
        'Arrival_hour': arrival_hour,
        'Arrival_minute': arrival_minute,
        'Duration_mins': duration_mins,
        'Total_Stops': encoded_stops
    }

    input_df = pd.DataFrame(np.zeros((1, len(features_list))), columns=features_list)

    # Fill basic features
    for feature, value in input_dict.items():
        if feature in input_df.columns:
            input_df.at[0, feature] = value

    # One-hot for airline and additional info
    airline_col = f"Airline_{airline}"
    if airline_col in input_df.columns:
        input_df.at[0, airline_col] = 1

    add_info_col = f"Additional_Info_{add_info}"
    if add_info_col in input_df.columns:
        input_df.at[0, add_info_col] = 1

    # Show preview of input data
    if st.checkbox("Show Model Input DataFrame"):
        st.write(input_df)

    # Predict
    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Flight Price: ₹{int(prediction):,}")

# --------------------
# Main App with Navigation
# --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Predict Price"])

if page == "EDA":
    eda_page()
else:
    prediction_page()
