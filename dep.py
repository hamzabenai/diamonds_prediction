import pandas as pd 
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

def load_data():
  data = pd.read_csv('train_data.csv')
  return data 

def load_model():
  file = 'model.joblib'
  model = joblib.load(file)
  return model


def predict(model, input_data):
  input_data = input_data.drop(columns=['price'])
  for column in input_data.columns:
    if input_data[column].dtype == 'object':
      le = LabelEncoder()
      input_data[column] = le.fit_transform(input_data[column])
  prediction = model.predict(input_data)
  return prediction

def main():
  st.title('This is a streamlit app for predicting the diamonds price:')
  st.write('This is a streamlit app for predicting the diamonds price:')
  st.write('here is the sample of the data :')
  data = load_data()
  st.write(data.head())
  st.title('We need you so submit the input data as csv file :')
  input_file = st.file_uploader("Upload CSV", type=["csv"])
  if input_file is not None:
    input_data = pd.read_csv(input_file)
    st.write('Here is the input data:')
    st.write(input_data.head())
    model = load_model()
    prediction = predict(model, input_data) # making the prediction
    st.write('Here is the prediction:')
    st.write(prediction)
    st.write('thanks')
if __name__ == '__main__':
  main()