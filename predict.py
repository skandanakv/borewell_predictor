# import joblib
# import numpy as np

# # Load the trained model
# model = joblib.load('groundwater_model.pkl')

# # Define sample inputs
# recharge_from_rain_monsoon = 4000
# recharge_other_sources_monsoon = 1200
# recharge_from_rain_non_monsoon = 800
# recharge_other_sources_non_monsoon = 300
# total_annual_gw_recharge = 25000  # ✅ You forgot this line earlier

# # Pack the inputs into a feature array
# features = np.array([[recharge_from_rain_monsoon,
#                       recharge_other_sources_monsoon,
#                       recharge_from_rain_non_monsoon,
#                       recharge_other_sources_non_monsoon,
#                       total_annual_gw_recharge]])

# # Make prediction
# predicted_extraction = model.predict(features)[0]

# # Prevent negative predictions (optional safety)
# predicted_extraction = max(predicted_extraction, 0)

# print(f"\n🔮 Predicted Total Annual Ground Water Extraction: {predicted_extraction:.2f}")

# # Borewell decision
# if predicted_extraction < 20000:
#     print("❌ Not suitable for Borewell Installation")
# else:
#     print("✅ Suitable for Borewell Installation")





# import joblib
# import pandas as pd

# # Load model
# model = joblib.load('groundwater_model.pkl')

# # Sample input (REALISTIC values)
# input_data = {
#     'Recharge from rainfall During Monsoon Season': [8000],
#     'Recharge from other sources During Monsoon Season': [40000],
#     'Recharge from rainfall During Non Monsoon Season': [3000],
#     'Recharge from other sources During Non Monsoon Season': [20000]
# }

# features_df = pd.DataFrame(input_data)

# # Predict
# predicted_extraction = model.predict(features_df)[0]
# predicted_extraction = max(predicted_extraction, 0)

# print(f"\n🔮 Predicted Total Annual Ground Water Extraction: {predicted_extraction:.2f}")
# if predicted_extraction < 20000:
#     print("❌ Not suitable for Borewell Installation")
# else:
#     print("✅ Suitable for Borewell Installation")






import joblib
import numpy as np

# Load model
model = joblib.load('groundwater_model.pkl')

# 6 Required Features (order must match training)
features = [
    'Total Annual Ground Water Recharge',
    'Total Natural Discharges',
    'Annual Extractable Ground Water Resource',
    'Current Annual Ground Water Extraction For Irrigation',
    'Current Annual Ground Water Extraction For Domestic & Industrial Use',
    'Net Ground Water Availability for future use'
]

# ✅ Sample Input (replace with real or interactive inputs)
input_data = {
    'Total Annual Ground Water Recharge': 100.0,
    'Total Natural Discharges': 15.0,
    'Annual Extractable Ground Water Resource': 85.0,
    'Current Annual Ground Water Extraction For Irrigation': 25.0,
    'Current Annual Ground Water Extraction For Domestic & Industrial Use': 10.0,
    'Net Ground Water Availability for future use': 50.0
}

# Convert to model input format
X_input = np.array([input_data[feature] for feature in features]).reshape(1, -1)

# Predict
predicted_extraction = model.predict(X_input)[0]

# Clip negative values to 0
predicted_extraction = max(predicted_extraction, 0)

# Output prediction
print(f"\n🔮 Predicted Total Current Annual Ground Water Extraction: {predicted_extraction:.2f}")

# ✅ Borewell Decision Logic
if predicted_extraction < 1000:
    print("✅ Borewell CAN be installed at this location.")
elif predicted_extraction < 5000:
    print("⚠️ Borewell MAY be installed — needs local approval.")
else:
    print("🚫 Borewell CANNOT be installed — groundwater stress too high.")
