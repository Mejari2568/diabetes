#import lib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pickle import dump
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("diabetes.csv")

#features and target
features = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
target = data["Outcome"]

#model
model = LogisticRegression()
model.fit(features, target)

#save the model
with open("dia.pkl", "wb") as f:
    dump(model, f)

print("Model created and saved as 'dia.pkl'")
