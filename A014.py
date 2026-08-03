# Generated from: MLOPS_PRACTICAL1.ipynb
# Converted at: 2026-07-13T05:15:22.174Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import json

df=pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1.csv")
df.head()

X=df[['Tower_ID','Temperature_C','Battery_Voltage','Power_Consumption_W','Signal_Strength_Percent','Fan_Speed_RPM','Humidity_Percent','Traffic_Load','Tower_Age_Years']]
y=df['Failure_Within_48Hrs']


X

y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print("Accuracy of random forest",accuracy)


joblib.dump(model,"telecom_tower_model.pkl")


metrics={"accuracy":accuracy}
with open("metrics.json","w") as f:
    json.dump(metrics,f,indent=4)
print("training Completed Successfully")
