from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    age = int(request.form["age"])

    features = np.array([[area, bedrooms, age]])
    prediction = model.predict(features)[0]

    return render_template("index.html", prediction=f"Estimated Price: ${int(prediction):,}")

if __name__ == "__main__":
    app.run(debug=True)
