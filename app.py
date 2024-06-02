import streamlit as st
import pymongo
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Function to connect to MongoDB and fetch data
def fetch_data():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["temperatureDB"]
    collection = db["temperatureCollection"]
    data_from_db = collection.find()
    df_from_db = pd.DataFrame(list(data_from_db))
    df_from_db.drop('_id', axis=1, inplace=True)
    return df_from_db

# Fetch the data from MongoDB
df = fetch_data()

# Load the saved model
model = joblib.load('temperature_model.pkl')

# Streamlit app
st.title("Temperature Conversion Model")
st.write("## Data Points")
st.write("Celsius calculation · = (Fahrenheit - 32) / 1.8")
st.write("Celsius calculation . = Kelvin - 273,15")
# Plot the data points
fig, ax = plt.subplots()
ax.scatter(df['Fahrenheit'], df['Celsius'], color='blue', label='Data Points')
ax.set_xlabel('Fahrenheit')
ax.set_ylabel('Celsius')
ax.legend()
st.pyplot(fig)

# Model details
st.write("## Model Details")
st.write(f"Coefficients: {model.coef_}")
st.write(f"Intercept: {model.intercept_}")

# Explain evaluation metrics
st.write("### Model Evaluation Metrics")
st.write("""
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors—that is, the average absolute difference between the estimated values and the actual value.
- **R-squared (R2 Score)**: Indicates how well data points fit a line or curve. The closer the R2 value is to 1, the better the fit.
""")

# Predict new data points
st.write("## Predict New Data")
fahrenheit = st.number_input('Enter Fahrenheit value')
kelvin = st.number_input('Enter Kelvin value')

if st.button('Predict'):
    new_data = np.array([[fahrenheit, kelvin]])
    prediction = model.predict(new_data)
    st.write(f"Predicted Celsius: {prediction[0]}")

    # Evaluate model with existing data
    X = df[['Fahrenheit', 'Kelvin']]
    y = df['Celsius']
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    st.write(f"### Evaluation Metrics on Existing Data")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R-squared: {r2}")

    # Plot the new data point
    fig, ax = plt.subplots()
    ax.scatter(df['Fahrenheit'], df['Celsius'], color='blue', label='Data Points')
    ax.scatter(fahrenheit, prediction[0], color='red', label='New Prediction')
    ax.set_xlabel('Fahrenheit')
    ax.set_ylabel('Celsius')
    ax.legend()
    st.pyplot(fig)
