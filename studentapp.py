import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit UI
def main():
    #st.title("Student Performance Prediction")
    st.write("<h1 style='text-align: center; color: blue;'>Student Performance Prediction</h1>", unsafe_allow_html=True)
    # Load dataset
    df = pd.read_csv("Student_Performance.csv")
    
    # Display dataset information
    #st.subheader("Dataset Preview")
    #st.write(df.head())
    
    # Encode categorical variable
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
    
    # Define features and target variable
    X = df.drop(columns=['Performance Index'])
    y = df['Performance Index']
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_lr = lr_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred_lr)
    mse = mean_squared_error(y_test, y_pred_lr)
    r2 = r2_score(y_test, y_pred_lr)
    
    #st.subheader("Model Performance")
    #st.write(f"Mean Absolute Error (MAE): {mae}")
    #st.write(f"Mean Squared Error (MSE): {mse}")
    #st.write(f"R2 Score: {r2}")
    
    # User input for prediction
    st.markdown("<h4 style='text-align: center; color: yellow;'>Predict Student Performance</h4>", unsafe_allow_html=True)
    hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=6)
    previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=80)
    extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=8)
    sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=100, value=5)
    
    # Encode user input
    extracurricular_encoded = 1 if extracurricular == "Yes" else 0
    user_data = [[hours_studied, previous_scores, extracurricular_encoded, sleep_hours, sample_papers]]
    user_data_scaled = scaler.transform(user_data)
    
    # Predict performance
    if st.button("Predict Performance"):
        prediction = lr_model.predict(user_data_scaled)[0]
        st.success(f"Predicted Performance Index: {prediction:.2f}")
    
if __name__ == "__main__":
    main()
