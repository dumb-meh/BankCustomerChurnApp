import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('Model_RF.pkl')

# Mapping of categorical variables
all_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Geography': {'France': 0, 'Germany': 1, 'Spain': 2},
    'Has Credit Card': {'Yes': 1, 'No': 0},
    'Is Active Member': {'Yes': 1, 'No': 0},
    'Number of Products': {'1': 1, '2': 2, '3': 3, '4': 4},
}
def prediction(CreditScore, age, Tenure, Balance, Salary, Gender, Geography, Credit_Card, Member, Products):
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
# Function to preprocess input data
def preprocess_input(data):
    encoder = LabelEncoder()
    data['Geography'] = encoder.fit_transform(data['Geography'])
    data['Gender'] = encoder.fit_transform(data['Gender'])
    return data

# Function to predict churn
def predict_churn(input_data):
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    return prediction

# Function to generate prediction and update dataframe
def generate_prediction_and_update_df(df):
    prediction = predict_churn(df)
    df['Prediction'] = prediction
    return df

# Function to download dataframe as CSV
def download_csv(df, filename='output.csv'):
    csv = df.to_csv(index=False)
    st.download_button(label='Download Predictions', data=csv, file_name=filename, mime='text/csv')

# Title and background image
st.set_page_config(page_title='Bank Customer Churn Prediction')
image = Image.open('cover.png')
st.image(image)
st.title('Churn Prediction')

# User input form
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

if st.button('Predict'):
    prediction = prediction(CreditScore, age, Tenure, Balance, Salary, Gender, Geography, Credit_Card, Member, Products)
    st.subheader('Prediction:')
    st.write(prediction)

# File uploader
st.header('Upload CSV File')
uploaded_file = st.file_uploader('Upload your CSV file here', type='csv')

if uploaded_file is not None:
    # Read uploaded CSV file into DataFrame
    df = pd.read_csv(uploaded_file)
    
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # Display uploaded DataFrame
    st.subheader('Uploaded DataFrame')
    st.write(df)

    # Perform prediction and update DataFrame
    df_with_predictions = generate_prediction_and_update_df(df)

    # Display DataFrame with predictions
    st.subheader('DataFrame with Predictions')
    st.write(df_with_predictions)

    # Download DataFrame as CSV
    download_csv(df_with_predictions, filename='predictions.csv')
