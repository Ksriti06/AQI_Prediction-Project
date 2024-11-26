import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set the page title
st.title("Air Quality Index (AQI) Prediction")

# Allow the user to upload a dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Select relevant columns for modeling
    selected_columns = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", 
        "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
    ]
    target_column = "AQI"

    if all(col in data.columns for col in selected_columns + [target_column]):
        # Filter data to keep only selected columns
        data_filtered = data[selected_columns + [target_column]]

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        data_imputed = imputer.fit_transform(data_filtered)
        data_cleaned = pd.DataFrame(data_imputed, columns=data_filtered.columns)

        # Define features (X) and target (y)
        X = data_cleaned[selected_columns]
        y = data_cleaned[target_column]

        # Add feature engineering
        X["PM_Ratio"] = X["PM2.5"] / (X["PM10"] + 1)
        X["NO_Ratio"] = X["NO"] / (X["NO2"] + 1)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # User selects the model
        st.subheader("Choose Machine Learning Model")
        model_choice = st.selectbox(
            "Select a model:",
            ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "Support Vector Machine", "K-Nearest Neighbors"]
        )

        # Initialize the selected model
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(random_state=42, n_estimators=100)
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_choice == "Support Vector Machine":
            model = SVR()
        elif model_choice == "K-Nearest Neighbors":
            model = KNeighborsRegressor()

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Predict on test data
        y_pred = model.predict(X_test_scaled)

        # Display evaluation metrics
        st.subheader("Model Performance")
        st.write(f"Selected Model: {model_choice}")
        st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

        # Visualization for Actual vs Predicted
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(y_test[:30])), y_test[:30], label='Actual', marker='o', color='blue')
        ax.plot(range(len(y_pred[:30])), y_pred[:30], label='Predicted', marker='x', color='red')
        ax.set_title(f"Actual vs Predicted AQI ({model_choice})")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("AQI")
        ax.legend()
        st.pyplot(fig)

        # Allow user to input AQI values for prediction
        st.subheader("Predict AQI for Custom Input")
        user_input = [st.number_input(f"Enter value for {col}", value=0.0) for col in selected_columns]
        if st.button("Predict AQI"):
            user_input_scaled = scaler.transform([user_input])
            predicted_aqi = model.predict(user_input_scaled)[0]
            st.write(f"Predicted AQI: {predicted_aqi:.2f}")
    else:
        st.error("The dataset does not contain the required columns.")
else:
    st.info("Please upload a dataset to proceed.")
