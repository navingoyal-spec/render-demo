import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pickled model
with open("decision_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scalar = pickle.load(f)


def predict_note_authentication(UserID, Gender, Age, EstimatedSalary):
    output = model.predict(scalar.transform([[Age, EstimatedSalary]]))
    prediction = output[0]
    return prediction


def main():

    html_temp = """
    <div style="background-color:blue;padding:10px">
        <center><p style="font-size:40px;color:white;">Poornima Institute of Engineering & Technology</p></center>
        <center><p style="font-size:30px;color:white;">Course on Machine Learning</p></center>
        <center><p style="font-size:25px;color:white;">Decision Tree ML Project Deployment</p></center>
        <center><p style="font-size:25px;color:white;">By:- Dr. Navin Kr. Goyal</p></center>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.header("Item Purchase Prediction")

    UserID = st.text_input("UserID", "")
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Age = st.number_input("Insert Age", min_value=5, max_value=150)
    EstimatedSalary = st.number_input("Insert Salary", min_value=1, max_value=1500000)

    if st.button("Predict"):
        result = predict_note_authentication(UserID, Gender, Age, EstimatedSalary)

        if result == 1:
            st.success("Model has predicted: Purchased ✅")
        else:
            st.warning("Model has predicted: Not Purchased ❌")

    if st.button("About"):
        st.subheader("Developed by Dr. Navin Kr. Goyal")
        st.subheader("Associate Professor")


if __name__ == "__main__":
    main()
