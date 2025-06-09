# # train_model.py

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import pickle

# # Load dataset
# df = pd.read_csv("data/borewell_dataset.csv")

# # Select relevant features
# features = [
#     'Recharge from rainfall During Monsoon Season',
#     'Recharge from other sources During Monsoon Season',
#     'Recharge from rainfall During Non Monsoon Season',
#     'Recharge from other sources During Non Monsoon Season',
#     'Annual Extractable Ground Water Resource',
#     'Stage of Ground Water Extraction (%)'
# ]
# target = 'Total Current Annual Ground Water Extraction'

# # Drop rows with missing values
# df = df[features + [target]].dropna()

# X = df[features]
# y = df[target]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f"\nModel trained successfully! MSE: {mse:.2f}")

# # ✅ Save the model (NOT predictions)
# with open("groundwater_model.pkl", "wb") as f:
#     pickle.dump(model, f)




import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and clean dataset
data = pd.read_csv('data/borewell_dataset.csv')
data.columns = data.columns.str.strip().str.replace('\n', ' ').str.replace(' +', ' ', regex=True)

print("🧾 Cleaned column names:")
for col in data.columns:
    print(f"- {col}")
# ✅ Define 6 features
features = [
    'Total Annual Ground Water Recharge',
    'Total Natural Discharges',
    'Annual Extractable Ground Water Resource',
    'Current Annual Ground Water Extraction For Irrigation',
    'Current Annual Ground Water Extraction For Domestic & Industrial Use',
    'Net Ground Water Availability for future use'
]


target = 'Total Current Annual Ground Water Extraction'

# ✅ Drop rows with NaNs in required columns
required_cols = features + [target]
data_cleaned = data.dropna(subset=required_cols)

# Extract features and labels
X = data_cleaned[features]
y = data_cleaned[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained successfully! MSE: {mse:.2f}")

# Save model
joblib.dump(model, 'groundwater_model.pkl')
print("📦 Model saved as 'groundwater_model.pkl'")

