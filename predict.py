import joblib
import pandas as pd
model = joblib.load("telecom_tower_model.pkl")

new_data = pd.DataFrame({
    'Tower_ID':[2],
    'Temperature_C':[43.9],
    'Battery_Voltage':[53.33],
    'Power_Consumption_W':[736],
    'Signal_Strength_Percent':[69],
    'Fan_Speed_RPM':[3139],
    'Humidity_Percent':[36],
    'Traffic_Load':[3427],
    'Tower_Age_Years':[6]
    })

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Hardware Failure")
else:
    print("Tower is Healthy")
