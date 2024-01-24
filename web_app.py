import pickle
import streamlit as st
import pandas as pd
import numpy as np

data = pd.read_csv('Cleaned_data.csv')
locations = sorted(data['location'].unique())
pipe_ridge = pickle.load(open('RidgeModel.pkl', 'rb'))


def predict(location, bedrooms, bath, square_ft):
    # Convert non-numeric features to float
    bedrooms = np.float64(bedrooms)
    bath = np.float64(bath)
    square_ft = square_ft

    # Create a DataFrame with the converted values
    input_data = pd.DataFrame([[location, square_ft, bath, bedrooms]],
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe_ridge.predict(input_data)[0] * 1e5
    return str(np.round(prediction, 2))


st.title("Bangalore House Price Prediction")

# Selectbox for location
selected_location = st.selectbox("Select your preferred location", locations)

# Input for number of bedrooms (bhk)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=16, step=1, value=2)

# Input for number of bathrooms
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=16, step=1, value=2)

# Input for total square footage
total_sqft = st.number_input("Total Square Footage", min_value=300, max_value=30400, step=100, value=1000)

# Button to submit the form
if st.button("Predict"):
    # Process the form data here (you can replace this with your own logic)
    result = (f"Submitted Data: \nNumber of Bathrooms: {num_bathrooms}\nTotal Square Footage: {total_sqft} sqft"
              f"\nNumber of Bedrooms (BHK): {bhk}\nSelected Location: {selected_location}")
    predicted = predict(selected_location, bhk, num_bathrooms, total_sqft)
    predicted_price = f"Predicted Price : Rs {predicted}"
    st.success(result)
    st.success(predicted_price)
