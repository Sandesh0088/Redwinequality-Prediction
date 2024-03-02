# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:53:41 2024

@author: ACER
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/ACER/Desktop/spyder/trained_model.sav', 'rb'))



# creating a function for Prediction

def winequality_prediction(input_data):
    

    input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
        print('Good Quality  Red Wine')
    else:
            print('Bad Quality Red Wine')
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    fixed_acidity = st.text_input('fixed acidity')
    volatile_acidity = st.text_input('volatile acidity')
    citric_acid= st.text_input('citric acid')
    residual_sugar = st.text_input('residual sugar')
    chlorides = st.text_input('chlorides')
    free_sulfur_dioxide = st.text_input('free sulfur dioxide')
    total_sulfur_dioxide= st.text_input('total sulfur dioxide	')
    density = st.text_input('density')
    pH = st.text_input('pH')
    sulphates = st.text_input('sulphates')
    alcohol = st.text_input('alcohol')
    
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Predict'):
        diagnosis = winequality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  