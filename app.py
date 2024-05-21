import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Load the trained model
model = pickle.load(open('Model_RF.pkl', 'rb'))

# Title
st.title('Churn Prediction')
image = Image.open('cover.png')
st.image(image, '')
# Function to predict churn
def predict_churn(CreditScore, age, Tenure, Balance, Salary, Gender, Geography, Credit_Card, Member, Products):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Age': [age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'EstimatedSalary': [Salary],
        'Gender': [Gender],
        'Geography': [Geography],
        'HasCrCard': [Credit_Card],
        'IsActiveMember': [Member],
        'NumOfProducts': [Products]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return prediction
    return "Yes, Customer will Churn" if prediction == 1 else "No, Customer will not Churn"

# Input form
st.header('Enter Customer Details')
CreditScore = st.slider('Credit Score', 0, 1000)
age = st.slider('Age', 18, 100)
Tenure = st.slider('Tenure', 0, 20)
Balance = st.slider('Balance', 0, 1000000)
Salary = st.slider('Salary', 0, 1000000)
Gender = st.selectbox('Gender', ['Male', 'Female'])
Geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
Credit_Card = st.selectbox('Has Credit Card', ['Yes', 'No'])
Member = st.selectbox('Is Active Member', ['Yes', 'No'])
Products = st.selectbox('Number of Products', ['1', '2', '3', '4'])

# Predict button
if st.button('Predict'):
    prediction = predict_churn(CreditScore, age, Tenure, Balance, Salary, Gender, Geography, Credit_Card, Member, Products)
    st.subheader('Prediction:')
    st.write(prediction)
