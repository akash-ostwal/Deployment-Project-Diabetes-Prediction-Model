import streamlit as st
import pandas as pd
import numpy as nop
import pickle

# set the tab=> page title
st.set_page_config(page_title="Diabetes Prediction WebPage",layout='wide')

# Set the page tile
st.title("Diabetes Prediction Project")
st.subheader("By Akash Ostwal")
st.subheader("Provide your test results below, and click on Predict button to view the Diabetes results")

# Take all the required inputs from the user
Gender = st.number_input('Gender (0 for Female and 1 foe Male)', min_value=0,max_value=1)
Age = st.number_input('Age',min_value=10,max_value=80)
Urea = st.number_input('Urea')
Cr = st.number_input('Creatinine')
HbA1c = st.number_input('Hb1A1c levels')
Chol = st.number_input('Cholestrol')
TG = st.number_input('Thyroid')
HDL = st.number_input('HD levels')
LDL = st.number_input('LDL levels')
VLDL = st.number_input('VLDL levels')
BMI = st.number_input('BMI')

# Provide a button for user to click and get the predictions
submit = st.button('predict the result here')

# Load the pickle files: preprocessor, model
with open('pre.pkl','rb') as file:
    pre = pickle.load(file)
with open ('model.pkl','rb') as file:
    model = pickle.load(file)

# What should happen when user clicks on submit button
if submit:
    dct = {
        'Gender':[Gender],
        'AGE':[Age],
        'Urea':[Urea],
        'Cr':[Cr],
        'HbA1c':[HbA1c],
        'Chol':[Chol],
        'TG':[TG],
        'HDL':[HDL],
        'LDL':[LDL],
        'VLDL':[VLDL],
        'BMI':[BMI]
    }
    xnew = pd.DataFrame(dct)
    xnew_pre = pre.transform(xnew)
    preds = model.predict(xnew_pre)
    if preds[0]==0:
        res_op = 'Non-Diabetic'
    elif preds[0]==1:
        res_op = 'Diabetic'
    else:
        res_op = 'Predict Diabetic'
    st.subheader(f"Your results are : {res_op}")