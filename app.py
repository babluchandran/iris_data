from flask import Flask, request, render_template
import joblib
import numpy as np

# naming our app as app
app = Flask(__name__)

# loading the pickle file for creating the web app
model = joblib.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 1:
        return render_template("index.html", prediction_text="The flower is setosa")
    elif output == 2:
        return render_template("index.html", prediction_text="The flower is virginica")
    else:
        return render_template("index.html", prediction_text="The flower is versicolor")


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
