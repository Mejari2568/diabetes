from flask import Flask, render_template, request
from pickle import load
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    msg = None
    if request.method == "POST":
        basedir = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(basedir, "dia.pkl")
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                model = load(f)

            try:
                pregnancies = float(request.form["pregnancies"])
                glucose = float(request.form["glucose"])
                blood_pressure = float(request.form["blood_pressure"])
                skin_thickness = float(request.form["skin_thickness"])
                insulin = float(request.form["insulin"])
                bmi = float(request.form["bmi"])
                dpf = float(request.form["dpf"])
                age = float(request.form["age"])

                data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
                result = model.predict(data)
                msg = "Diabetes Positive" if result[0] == 1 else "Diabetes Negative"
            except Exception as e:
                msg = f"Error: {str(e)}"
        else:
            msg = fn + " does not exist"

    return render_template("home.html", msg=msg)

if __name__ == "__main__":
    app.run(debug=True)
