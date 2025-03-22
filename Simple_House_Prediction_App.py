import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

# Title and description
st.title("🏡 House Price Prediction")
st.write("🔹 Predict the price of a house based on its size and number of bedrooms using Machine Learning.")

# Example Data: House Size (sq ft), Bedrooms, and Prices
sizes = np.array([800, 1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
bedrooms = np.array([2, 3, 3, 2, 4, 3]).reshape(-1, 1)
prices = np.array([2000000, 2500000, 3000000, 3500000, 4000000, 4500000])

# Combine Features (Size & Bedrooms)
features = np.column_stack((sizes, bedrooms))

# Train the Model
model = LinearRegression()
model.fit(features, prices)

# Sidebar Input
st.sidebar.header("📌 Enter House Details")
size = st.sidebar.number_input("🏠 House Size (sq ft):", min_value=500, max_value=5000, step=50, value=1500)
num_bedrooms = st.sidebar.slider("🛏 Number of Bedrooms:", min_value=1, max_value=6, value=3)

# Prediction
if size and num_bedrooms:
    prediction_value = np.array([[size, num_bedrooms]])
    prediction_price = model.predict(prediction_value)
    st.subheader("💰 Predicted House Price")
    st.write(f"🏡 Estimated price for a {size} sq ft house with {num_bedrooms} bedrooms: **₹{prediction_price[0]:,.2f}**")

# Show Model Formula (Optional)
st.caption(f"📈 **Price = {model.coef_[0]:.2f} × Size + {model.coef_[1]:.2f} × Bedrooms + {model.intercept_:.2f}**")
