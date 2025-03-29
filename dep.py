import pandas as pd 
import streamlit as st
import joblib
import os

def load_data():
    data = pd.read_csv('train_data.csv')
    return data 

def load_model():
    file = 'model.joblib'
    try:
        model = joblib.load(file)
        # Verify it's actually a model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid model (missing predict method)")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def predict(model, input_data):
    # Safely drop price column if it exists
    if 'price' in input_data.columns:
        input_data = input_data.drop(columns=['price'])
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()

def main():
    st.title('Diamond Price Prediction App')
    
    # Show sample data
    st.write('Sample training data:')
    try:
        data = load_data()
        st.write(data.head())
    except Exception as e:
        st.error(f"Failed to load sample data: {str(e)}")
    
    # File uploader
    st.subheader('Upload prediction data')
    input_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if input_file is not None:
        try:
            input_data = pd.read_csv(input_file)
            st.write('Input data preview:')
            st.write(input_data.head())
            
            # Load model with verification
            model = load_model()
            st.success("Model loaded successfully!")
            
            # Make prediction
            prediction = predict(model, input_data)
            
            # Display results
            st.subheader('Prediction Results')
            st.write(prediction)
            
            # Optional: Show input data with predictions
            result = input_data.copy()
            result['Predicted Price'] = prediction
            st.write(result)
            
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")

if __name__ == '__main__':
    main()