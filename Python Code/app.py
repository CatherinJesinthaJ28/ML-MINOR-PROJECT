import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Apple Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Apple Stock Price Prediction")
st.markdown("### Machine Learning Minor Project")
st.write("This project predicts Apple's closing stock price using the Linear Regression algorithm.")

# Load Dataset
df = pd.read_csv("Dataset/archive/AAPL.csv")

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Total Features", len(df.columns))

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]

if st.button("🚀 Run Prediction"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_test.values[:100], label="Actual Price")
    ax.plot(y_pred[:100], label="Predicted Price")
    ax.set_title("Actual vs Predicted Apple Stock Price")
    ax.set_xlabel("Days")
    ax.set_ylabel("Closing Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.success("Prediction completed successfully! ✅")

st.markdown("---")
st.markdown("### 👩‍💻 Developed by")
st.write("**Catherin Jesintha J**")
st.write("B.Sc. Computer Science")
st.write("St. Anne's Arts and Science College, Chennai")