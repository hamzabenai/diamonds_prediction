import pandas as pd 
import streamlit as st
import joblib

def load_data():
  data = pd.read_csv('train_data.csv')
  return data 

def load_model():
  with open('model.joblib', 'rb') as file:
    model = joblib.load(file)
  return model


def predict(model, input_data):
  prediction = model.predict(input_data)
  return prediction

def main():
  st.title('This is a streamlit app for predicting the diamonds price:')
  st.write('This is a streamlit app for predicting the diamonds price:')
  st.write('here is the sample of the data :')
  data = load_data()
  st.write(data.head())
  
if __name__ == '__main__':
  main()