import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Item Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("decision_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scalar = pickle.load(f)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.result-success {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 10px;
    color: #2e7d32;
    font-size: 20px;
    text-align: center;
}
.result-fail {
    background-color: #ffebee;
    padding: 20px;
    border-radius: 10px;
    color: #c62828;
    font-size: 20px;
    text-align: center;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h2>Poornima Institute of Engineering & Technology</h2>
    <h3>Department of Computer Engineering</h3>
    <h4>Course on Machine Learning</h4>
    <h4>Decision Tree ML Project Deployment</h4>
    <p><b>By: Dr. Navin Kr. Goyal</b></p>
</div>
""", unsafe_allow_html=True)

#st.write("")

# ---------------- PREDICTION FUNCTION ----------------
def predict_note_authentication(Gender,Age, EstimatedSalary):
    if(Gender=='Male'):
        Gender=1
    else:
        Gender=0
    output = model.predict(scalar.transform([[Gender,Age, EstimatedSalary]]))
    return output[0]

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 User Information")

UserID = st.sidebar.text_input("User ID")
Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Age = st.sidebar.number_input("Age", min_value=5, max_value=150, step=1)
EstimatedSalary = st.sidebar.number_input(
    "Estimated Salary (₹)",
    min_value=1,
    max_value=1500000,
    step=1000
)

predict_btn = st.sidebar.button("🔮 Predict")
about_btn = st.sidebar.button("ℹ️ About")

# ---------------- MAIN CONTENT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Item Purchase Prediction System")
    st.write("""
    This application predicts whether a user is likely to purchase an item
    based on **Age** and **Estimated Salary**, using a **Decision Tree Machine Learning model**.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if predict_btn:
        result = predict_note_authentication(Gender,Age, EstimatedSalary)

        if result == 1:
            st.markdown("""
            <div class="result-success">
                ✅ User is likely to purchase the item
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-fail">
                ❌ User is not likely to purchase the item
            </div>
            """, unsafe_allow_html=True)

# ---------------- ABOUT SECTION ----------------
if about_btn:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👨‍🏫 About the Developer")
    st.write("""
    **Dr. Navin Kr. Goyal**  
    Associate Professor  
    Department of Computer Science & Engineering  

    **Specialization:**  
    - Machine Learning  
    - Data Analytics  
    - Artificial Intelligence  

    This project is developed for **academic learning and deployment demonstration**.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    © 2026 | Machine Learning Project | PIET Jaipur
</div>
""", unsafe_allow_html=True)
