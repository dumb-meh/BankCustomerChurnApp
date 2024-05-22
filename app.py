import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder


model = joblib.load('Model_RF.pkl')

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

def preprocess_input(data):
    encoder = LabelEncoder()
    data['Geography'] = encoder.fit_transform(data['Geography'])
    data['Gender'] = encoder.fit_transform(data['Gender'])
    return data

def predict_churn(input_data):
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    return prediction

def generate_prediction_and_update_df(df):
    predictions = predict_churn(df)
    df['Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
    return df


def download_csv(df, filename='output.csv'):
    csv = df.to_csv(index=False)
    st.download_button(label='Download Predictions', data=csv, file_name=filename, mime='text/csv')


st.set_page_config(page_title='Bank Customer Churn Prediction')
image = Image.open('cover.png')
st.image(image)
st.title('Churn Prediction')

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

st.header('Upload CSV File')
uploaded_file = st.file_uploader('Upload your CSV file here', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    st.subheader('Uploaded DataFrame')
    st.write(df)

    df_with_predictions = generate_prediction_and_update_df(df)


    st.subheader('DataFrame with Predictions')
    st.write(df_with_predictions)

    download_csv(df_with_predictions, filename='predictions.csv')
