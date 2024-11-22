# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:17:56 2024

@author: Lavanya N Khushi
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open("C:/Users/Lavanya N Khushi/Downloads/Diabetes/trained_model.sav", 'rb'))
def prediction(data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
def main():
    st.title("Diabetes Prediction")
    pregnancies=st.text_input("Number of pregnancies")
    glucose=st.text_input("Glucose level")
    bloodP=st.text_input("Blood Pressure")
    skinT=st.text_input("Skin Thickness")
    insulin=st.text_input("Insulin level")
    bmi=st.text_input("Body Mass Index(BMI)")
    dpf=st.text_input("Diabetes Pedigree Function")
    age=st.text_input("Age")
    
    outcome=''
    if st.button("Diabetes Test Result"):
        outcome=prediction([pregnancies,glucose,bloodP,skinT,insulin,bmi,dpf,age])
     
    st.success(outcome)
if __name__=='__main__':
    main()
    
        
        
        
        