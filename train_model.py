import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(42)

data = pd.DataFrame({
    "Temperature": np.random.normal(75, 10, 500),
    "Pressure": np.random.normal(30, 5, 500),
    "Vibration": np.random.normal(5, 1, 500),
    "Runtime_Hours": np.random.normal(2000, 300, 500)
})

# Failure condition
data["Failure"] = np.where(
    (data["Temperature"] > 85) |
    (data["Vibration"] > 6.5) |
    (data["Pressure"] > 40), 1, 0
)

# Split data
X = data.drop("Failure", axis=1)
y = data["Failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/predictive_model.pkl")

# Accuracy
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))
