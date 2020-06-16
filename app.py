# Importing essential libraries
# from flask import Flask, render_template, request
import pickle
import numpy as np
import streamlit as st 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
# Load the Random Forest CLassifier model
rfmodel = pickle.load(open('rmodel.pkl', 'rb'))
powertrans = pickle.load(open('powertransAD.pkl', 'rb'))

# app = Flask(__name__)

# @app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict', methods=['POST'])
def predict(age, workclass, fnlwgt, education, education_num, Marital_Status, occupation,         relationship,race,Capital_gain,Capital_loss,hours_per_week,native_country):
    prediction=rfmodel.predict([[age, workclass, fnlwgt, education, education_num, Marital_Status, occupation,         relationship,race,Capital_gain,Capital_loss,hours_per_week,native_country]])
    print(prediction)
    return prediction

def main():
    st.title("Adult Income Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Adult Income Analysis ML App </h2>
    </div>"""
    
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("age","")
    workclass = st.text_input("workclass","")
    fnlwgt = st.text_input("fnlwgt","")
    education = st.text_input("education","")
    education_num = st.text_input("education_num","")
    Marital_Status = st.text_input("Marital_Status","")
    occupation = st.text_input("occupation","")
    relationship = st.text_input("relationship","")
    race = st.text_input("race","")
    Capital_gain = st.text_input("Capital_gain","")
    Capital_loss = st.text_input("Capital_loss","")
    hours_per_week = st.text_input("hours_per_week","")
    native_country = st.text_input("native_country","")
    
    result=""
    if st.button("Predict"):
        result=predict(age,workclass, fnlwgt, education, education_num, Marital_Status, occupation,         relationship,race,Capital_gain,Capital_loss,hours_per_week,native_country)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()