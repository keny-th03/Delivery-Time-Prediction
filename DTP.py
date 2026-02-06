from flask import Flask, request, render_template
import pandas as pd  
import numpy as np 
import pickle

app = Flask(__name__)

#Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    price = float(request.form["price"])  
    freight_value = float(request.form["freight_value"])
    review_score = float(request.form["review_score"])
    purchase_weekday = int(request.form["purchase_weekday"])
    
    features = np.array([[price, freight_value, review_score, purchase_weekday]])
    
    prediction = model.predict(features)[0]
    
    return render_template(
        "result.html",
        prediction=round(float(prediction),2)
    )
    
@app.route("/visuals")
def visuals():
    return render_template("visuals.html")

if __name__ == "__main__":
    app.run(debug=True)
    


