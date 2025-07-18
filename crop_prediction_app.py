import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Loading the columns
load_col = joblib.load(open("columns.pkl", "rb"))

# Loading the saved model(pipeline)
loaded_model = joblib.load(open("crop_prediction.pkl", "rb"))


# Creating a function forr prediction
def model_prediction(input_data):
    # Define input data
    input_data = ("Albania","Maize",1990,1485.0,121.0,16.37)

    # Converting to a pandas dataframe
    input_df = pd.DataFrame([input_data], columns=load_col)

    # Making predictions
    pred = loaded_model.predict(input_df)
    return pred[0]
    
def main():
    # Giving a title for the app
    st.title("Crop Prediction Web App")
    
    # Getting input from the user
    Area = st.text_input("Country: ")
    Item = st.text_input("Crop: ")
    Year = st.text_input("Year: ")
    Average_rainfall = st.text_input("Average rain fall(mm) per year: ")
    Pesticides = st.text_input("Pesticides (tonnes): ")
    Average_temp = st.text_input("Average temperature: ")

    # Code for prediction
    crop_yield = ""
    
    # Creating a button to generate the predictive result
    if st.button("Crop Prediction result"):
        crop_yield = model_prediction([Area, Item, Year, Average_rainfall, Pesticides, Average_temp])
        st.success(crop_yield)
if __name__ == "__main__":
    main()