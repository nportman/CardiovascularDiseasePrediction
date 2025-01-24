import streamlit as st
import pandas as pd
from models.simple_model import predict_rule_based


def interactive_prediction():
    st.write("Enter the features for prediction:")
    
    slope = st.selectbox("Slope", [0, 1, 2, 3])
    
    if st.button("Predict"):
        feature_vector = pd.Series({'slope': slope})
        prediction = predict_rule_based(feature_vector)
        
        if prediction is not None:
            if prediction == 0.0:
                st.success("Prediction: Patient has no cardiovascular disease.")
            else:
                st.success("Prediction: Patient has cardiovascular disease.")
        else:
            st.error("Invalid slope value. Please enter a value between 0 and 3.")

def upload_and_predict():
    st.write("Upload a CSV file for batch predictions:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'slope' not in df.columns:
            st.error("The uploaded CSV must contain a 'slope' column.")
            return
        
        predictions = df['slope'].apply(lambda x: predict_rule_based(pd.Series({'slope': x})))
        df['prediction'] = predictions
        
        st.write("Predictions:")
        st.dataframe(df)
        
        # Option to download the results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

def main():
    st.title("Cardiovascular Disease Prediction")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        interactive_prediction()
    
    with tab2:
        upload_and_predict()

if __name__ == "__main__":
    main()