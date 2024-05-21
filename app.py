import streamlit as st
import joblib
import pandas as pd
from PIL import Image

model = joblib.load('Model_RF.pkl')

# Mapping of categorical variables
all_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Geography': {'France': 0, 'Germany': 1, 'Spain': 2},
    'Has Credit Card': {'Yes': 1, 'No': 0},
    'Is Active Member': {'Yes': 1, 'No': 0},
    'Number of Products': {'1': 1, '2': 2, '3': 3, '4': 4},
}


st.set_page_config(page_title='Bank Customer Churn Prediction')

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://miro.medium.com/v2/resize:fit:720/format:webp/0*8Iu_eymr6eR-YuQw");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title('Churn Prediction')

# Your Streamlit app code goes here


# Function to predict churn
def predict_churn(CreditScore, age, Tenure, Balance, Salary, Gender, Geography, Credit_Card, Member, Products):
    # Map string values to numerical values
    Gender = all_mappings['Gender'][Gender]
    Geography = all_mappings['Geography'][Geography]
    Credit_Card = all_mappings['Has Credit Card'][Credit_Card]
    Member = all_mappings['Is Active Member'][Member]
    Products = all_mappings['Number of Products'][Products]

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
    return "Yes, Customer will Churn" if prediction == 1 else "No, Customer will not Churn"

# Input form
st.header('Enter Customer Details')
CreditScore = st.slider('Credit Score', 0, 1000)
age = st.slider('Age', 18, 100)
Tenure = st.slider('Tenure', 0, 20)
Balance = st.number_input('Balance', value=0.0, step=1000.0)
Salary = st.number_input('Salary', value=0.0, step=1000.0)
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
