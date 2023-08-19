import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import openpyxl
import base64  # Add this line to import the base64 module

# Load the trained models
loaded_model = joblib.load(r"C:\Users\50510\Desktop\MAR IHTOYJ\Accident_Source_model.pkl")
loaded_Coverage_Code = joblib.load(r"C:\Users\50510\Desktop\MAR IHTOYJ\Coverage_Code_model.pkl")

# Load the dictionary from the file
with open(r'C:\Users\50510\Desktop\MAR IHTOYJ\numerical_to_code_mapping.pkl', 'rb') as f:
    numerical_to_code_mapping = pickle.load(f)

with open(r'C:\Users\50510\Desktop\MAR IHTOYJ\numerical_to_accident_mapping.pkl', 'rb') as f:
    numerical_to_accident_mapping = pickle.load(f)

# For Fitting vectorizer
import pandas as pd
import numpy as np
data = pd.read_excel(r"C:\Users\50510\Downloads\Dataset_Public.xlsx")
data = data[['Claim Description']]
# Clean the Data
data['Claim Description'] = data['Claim Description'].astype('str')
def clean(text):
        text = text.lower()
        return text
        
data['Claim Description'] = data['Claim Description'].apply(lambda x: clean(x))
data = data[0:10000]       

# Load the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(data['Claim Description'])


# Define the Streamlit app
def main():
    st.title("Accident Source and Coverage Code Prediction")

    # Upload the dataset
    uploaded_file = st.file_uploader("Upload a dataset excel file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        # Clean the Data
        df['Claim Description'] = df['Claim Description'].astype('str')

        def clean(text):
            text = text.lower()
            return text

        df['Claim Description'] = df['Claim Description'].apply(lambda x: clean(x))

        # Run the models on the uploaded dataset
        st.header("Predictions")
        st.subheader("Accident Source Predictions:")
        accident_predictions = predict_accident_source(df)
        st.write(accident_predictions)

        st.subheader("Coverage Code Predictions:")
        coverage_predictions = predict_coverage_code(df)
        st.write(coverage_predictions)

        # Display evaluation results and save to Excel
        st.header("Evaluation Results")
        # Display the evaluation results here

        # Save predictions to an Excel file
        excel_bytes = save_results_to_excel(accident_predictions, coverage_predictions)

        # Display download links for Excel files
        st.header("Download Prediction Results")
        st.write("Click below to download the prediction results:")

        # Download link for Accident Source predictions
        accident_link = get_download_link(excel_bytes, "Accident_Source_Predictions.xlsx")
        st.markdown(accident_link, unsafe_allow_html=True)

        # Download link for Coverage Code predictions
        coverage_link = get_download_link(excel_bytes, "Coverage_Code_Predictions.xlsx")
        st.markdown(coverage_link, unsafe_allow_html=True)

def predict_accident_source(data):
    # Process the data and get the predictions
    text_data = data["Claim Description"]  # Assuming "Claim Description" is the column name
    processed_data = vectorizer.transform(text_data)
    processed_data = processed_data.toarray()
    predictions = loaded_model.predict(processed_data)
    predicted_sources = [numerical_to_accident_mapping[label] for label in predictions]
    
    result_df = pd.DataFrame({"Claim Description": text_data, "Predicted_Source": predicted_sources})
    return result_df

def predict_coverage_code(data):
    text_data = data["Claim Description"]  
    processed_data = vectorizer.transform(text_data)
    processed_data = processed_data.toarray()
    predictions = loaded_Coverage_Code.predict(processed_data)
    predicted_sources = [numerical_to_code_mapping[label] for label in predictions]
    
    result_df = pd.DataFrame({"Claim Description": text_data, "Coverage Code": predicted_sources})
    return result_df

def save_results_to_excel(accident_df, coverage_df):
    with pd.ExcelWriter("prediction_results.xlsx") as writer:
        accident_df.to_excel(writer, sheet_name="Accident_Source_Predictions", index=False)
        coverage_df.to_excel(writer, sheet_name="Coverage_Code_Predictions", index=False)

    # Convert Excel file to bytes
    with open("prediction_results.xlsx", "rb") as f:
        excel_bytes = f.read()

    return excel_bytes

def get_download_link(file_bytes, file_name):
    # Generate a download link
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_bytes).decode()}" download="{file_name}">Click here to download</a>'
    return href



if __name__ == "__main__":
    main()
