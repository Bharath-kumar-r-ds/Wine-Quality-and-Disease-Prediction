import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    print(features)
    if features[-1][-1]==1:
        k_model = pickle.load(open("kwine_model.pkl", "rb"))
    prediction = k_model.predict((features[-1][:-1]).reshape(1,-1))
    if prediction[0]==1:
        return render_template("index.html", prediction_text = "The Quality of the wine is Good.")
    return render_template("index.html", prediction_text = "The Quality of the wine is Not Good.")

if __name__ == "__main__":
    flask_app.run(debug=True)