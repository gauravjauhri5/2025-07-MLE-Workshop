import pickle
from flask import Flask, request, jsonify

model_path = "./models/2022-01.pkl"
with open(model_path, "rb") as f_in:
    model = pickle.load(f_in)

# trip = {
#     "PULocationID": "43",
#     "DOLocationID": "238",
#     "trip_distance": 1.16,
# }

# prediction = model.predict(trip)
# print(prediction[0])


# "feature engineering"
def prepare_features(ride):
    features = dict(
        PULocationID=str(ride["PULocationID"]),
        DOLocationID=str(ride["DOLocationID"]),
        trip_distance=float(ride["trip_distance"]),
    )
    return features


def predict(features):
    prediction = model.predict(features)
    return float(prediction[0])


app = Flask("duration-prediction")


@app.route("/", methods=["GET"])
def home():
    return "hello world"


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    prediction = predict(features)
    result = {
        "prediction": {
            "duration": prediction,
        },
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)