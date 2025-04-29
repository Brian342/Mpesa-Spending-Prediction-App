from flask import Flask, request, render_template, jsonify
from flask_restful import Resource, Api
import pickle
import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util import encode_categorical_columns, encode_categorical_columns_training_encoder

model_path = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/Mpesa_LinearRegression.pkl"
scaling_type = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/Scaler.pkl"
encoding_type = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/df_encoding.pkl"

try:
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaling_type, "rb"))
    encode = pickle.load(open(encoding_type, "rb"))
except Exception as e:
    raise RuntimeError(f"Failed to load model, scaler, encode {e}")

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index2.html")


# CORS(app)

# api = Api(app)


@app.route('/predict', method=['POST'])
# class Prediction(Resource):
def predict():
    try:

        # data = request.get_json()
        # transaction_amount = int(data['transactionAmount'])
        # transaction_type = data['transactionType']
        # transaction_party = data['transactionParty']
        # paid_in_or_withdrawal = data['PaidInOrWithdrawal']

        data = request.form

        features = [
            'transaction_day', 'Year', 'Month', 'Date', 'Weekday', 'Hour', 'Minute',
            'Seconds', 'Transaction_type', 'Transaction_party', 'Transaction_amount',
            'paid_in_or_Withdraw', 'Balance'
        ]

        # dataframe creation
        input_df = pd.DataFrame([[data[field] for field in features]], columns=features)

        # Encode input data
        encoded_input_df = encode_categorical_columns(input_df, encoding_type='label')

        prediction = model.predict(encoded_input_df)
        prediction = int(prediction[0])

        return {"Predicted_spending": prediction}, 200  # Returning JSON response

    except Exception as e:
        print(f"Error: {e}")
        return {"error": "An error occurred while processing the prediction."}, 500


@app.route('/predict_api', methods=['POST'])
def predict_api() -> Union[str, jsonify]:
    try:
        data = request.get_json()

        feature = [
            'transaction_day', 'Year', 'Month', 'Date', 'Weekday', 'Hour', 'Minute',
            'Seconds', 'Transaction_type', 'Transaction_party', 'Transaction_amount',
            'paid_in_or_Withdraw', 'Balance'
        ]
        input_df = pd.DataFrame([[data[field] for field in feature]], columns=feature)

        encoded_input_df = encode_categorical_columns(input_df, encoding_type='label')

        prediction = model.predict(encoded_input_df)
        prediction = int(prediction[0])

        return jsonify({"result": prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
