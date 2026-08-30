import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score

model = joblib.load("telecom_tower_model.pkl")
print("Model Loaded Sucessfully")

new_data = pd.read_excel("new_tower_telemetry.xlsx")
print("New Production data loaded")

features = [
    "Tower_ID",
    "Temperature_C",
    "Battery_Voltage",
    "Power_Consumption_W",
    "Signal_Strength_Percent",
    "Fan_Speed_RPM",
    "Humidity_Percent",
    "Traffic_Load",
    "Tower_Age_Years"
]

x_new = new_data[features]

y_new = new_data["Failure_Within_48Hrs"]

predictions = model.predict(x_new)

accuracy = accuracy_score(
    y_new,
    predictions
    )

print("-----------------------------------")
print("MODEL DRIFT MONITORING")
print("-----------------------------------")
print(
    "Production Accuracy:",
    round(accuracy, 2)
)


mlflow.set_experiment(
     "Telecom_Tower_Model_Monitoring"
)

with mlflow.start_run():
    mlflow.log_param(
        "model",
        "Random Forest"
        )
    mlflow.log_param(
        "n_estimators",
        100
        )

    mlflow.log_metric(
        "production_accuracy",
        accuracy
    )
    print("Results logged successfully to MLflow")


threshold = 0.80
if accuracy < threshold:
    print()
    print("WARNING: Model performance has degraded!")
    print("Model retraining is recommended.")
else:
    print()
    print("Model performance is stable.")









