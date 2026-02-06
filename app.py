import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
#st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("decision_model.pkl","rb")
model=pickle.load(pickle_in)
pickle_in2 = open("scalar.pkl","rb")
scalar=pickle.load(pickle_in2)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model.predict(scalar.transform(([[Age,EstimatedSalary]])))
  print("Purchased", output)
  if output==[1]:
    prediction="Item will be purchased"
  else:
    prediction="Item will not be purchased"
  print(prediction)
  return prediction
def main():

    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institue of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Course on Machine Learning</p></center>
   <center><p style="font-size:25px;color:white;margin-top:10px;"> Decision Tree ML Project Deployment</p></center>
   <center><p style="font-size:25px;color:white;margin-top:10px;"> By:- Dr. Navin Kr. Goyal</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction")
    UserID = st.text_input("UserID","")
    Gender = st.selectbox('Gender',('Male', 'Female'))
    Age = st.number_input("Insert Age",5,150)
    EstimatedSalary = st.number_input("Insert salary",1,1500000)
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(UserID, Gender,Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Dr. Navin Kr. Goyal")
      st.subheader("Associate Professor")

if __name__=='__main__':
  main()

