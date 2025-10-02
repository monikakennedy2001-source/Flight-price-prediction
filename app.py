import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# --- Load artifacts ---
@st.cache_data
def load_artifacts():
    base_path = "c:/py project/Flight price prediction"
    with open(os.path.join(base_path, "final_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, "label_encoder_total_stops.pkl"), "rb") as f:
        le_total_stops = pickle.load(f)
    with open(os.path.join(base_path, "features.pkl"), "rb") as f:
        features_list = pickle.load(f)
    return model, le_total_stops, features_list

model, le_total_stops, features_list = load_artifacts()

# --- Streamlit UI ---
st.title("Flight Price Prediction")

# User inputs
total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops'])
journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=1)
journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=1)
dep_hour = st.slider("Departure Hour", 0, 23, 12)
dep_minute = st.slider("Departure Minute", 0, 59, 0)
arrival_hour = st.slider("Arrival Hour", 0, 23, 12)
arrival_minute = st.slider("Arrival Minute", 0, 59, 0)
duration_mins = st.number_input("Duration (minutes)", min_value=0, value=60)

# Encode total stops
encoded_stops = le_total_stops.transform([total_stops])[0]

# Create input dictionary for features
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

# For one-hot encoded columns (Airline, Source, Destination, Additional_Info), set 0 by default
# User can add selectboxes for these if you want (recommended for better predictions)

# Initialize DataFrame with zeros for all features
input_df = pd.DataFrame(np.zeros((1, len(features_list))), columns=features_list)

# Fill numeric and label encoded features
for feature, value in input_dict.items():
    if feature in input_df.columns:
        input_df.at[0, feature] = value

# Example: If you want user to select Airline, Source, Destination, Additional_Info,
# then you can update those dummy columns here:
# For example,
# airline = st.selectbox("Airline", [...list of airlines...])
# dummy_col = "Airline_" + airline
# if dummy_col in input_df.columns:
#     input_df.at[0, dummy_col] = 1

# --- Predict ---
if st.button("Predict Price"):
    pred_price = model.predict(input_df)[0]
    st.success(f"Estimated Flight Price: ₹{int(pred_price)}")
