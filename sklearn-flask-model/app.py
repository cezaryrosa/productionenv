import joblib
from flask import Flask, request, jsonify

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def print():
    return("Hello World!")

@app.route('/predict', methods=['GET'])
def predict():
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))

    features = [sepal_length, petal_length]
    prediction = int(model.predict(features))
    #predicted_class = int(model.predict(features))
    return jsonify(features=features,
    predicted_class=prediction)

if __name__ == '__main__':
    app.run(port=3333,host='0.0.0.0')