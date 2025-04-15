from pickle import load
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

fn = "dia.pkl"

if os.path.exists(fn):
    with open(fn, "rb") as f:
        model = load(f)

    # Input data
    pregnancies = float(input("Enter number of Pregnancies: "))
    glucose = float(input("Enter Glucose level: "))
    blood_pressure = float(input("Enter Blood Pressure: "))
    skin_thickness = float(input("Enter Skin Thickness: "))
    insulin = float(input("Enter Insulin level: "))
    bmi = float(input("Enter BMI: "))
    dpf = float(input("Enter Diabetes Pedigree Function: "))
    age = float(input("Enter Age: "))

    # Prepare input data for prediction
    d = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

    result = model.predict(d)

    if result[0] == 1:
        print("The model predicts that the person has diabetes.")
    else:
        print("The model predicts that the person does not have diabetes.")
else:
    print(fn, " does not exist")
