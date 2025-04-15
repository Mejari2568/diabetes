#import lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("diabetes.csv")
print(data)

#check for null data
print(data.isnull().sum())

#feature and target
features = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
target = data["Outcome"]

#train and test
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print(data['Outcome'].value_counts())

#model
model = LogisticRegression()
model.fit(x_train, y_train)

#performance
cr = classification_report(y_test, model.predict(x_test))
print(cr)
