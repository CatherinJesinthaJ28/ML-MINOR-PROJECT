import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Dataset/archive/AAPL.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Features and Target
X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Actual Price", color="blue")
plt.plot(y_pred[:100], label="Predicted Price", color="red")
plt.title("Actual vs Predicted Apple Stock Price")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()