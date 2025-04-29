from flask import Flask, request, render_template
from flask_restful import Resource, Api
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
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


CORS(app)

api = Api(app)


class Prediction(Resource):
    def post(self):
        try:
            data = request.get_json()
            transaction_amount = int(data['transactionAmount'])
            transaction_type = data['transactionType']
            transaction_party = data['transactionParty']
            paid_in_or_withdrawal = data['PaidInOrWithdrawal']

            features = {
                'transaction_day': 15,
                'Year': 2025,
                'Month': 4,
                'Date': 15,
                'Weekday': 1,
                'Hour': 14,
                'Minute': 30,
                'Seconds': 0,
                'Transaction_type': transaction_type,
                'Transaction_party': transaction_party,
                'Transaction_amount': transaction_amount,
                'paid_in_or_Withdraw': paid_in_or_withdrawal,
                'Balance': 1000
            }

            df = pd.DataFrame([features])

            if 'Transaction_party' in df.columns:
                df_encoding = encode.transform(df[['Transaction_party']])
                df_encoding = pd.DataFrame(df_encoding, columns=['Transaction_party'])
            else:
                df_encoding = pd.DataFrame({'Transaction_party': ['Unknown']})  # default placeholder

            df_scaling = scaler.transform(df.drop(['Transaction_party'], axis=1, errors='ignore'))
            df_encoded_scaling = pd.concat([df_encoding, pd.DataFrame(df_scaling)], axis=1)

            # Encode categorical columns
            for col in ['Transaction_type', 'Transaction_party', 'paid_in_or_Withdraw']:
                if col in df.columns:
                    df[col] = encode.transform(df[[col]])

            # Scale entire dataframe
            df_scaled = scaler.transform(df)

            # Predict
            predicted_spending = model.predict(df_scaled)
            predicted_spending = int(predicted_spending[0])

            return {"Predicted_spending": predicted_spending}, 200  # Returning JSON response

        except Exception as e:
            print(f"Error: {e}")
            return {"error": "An error occurred while processing the prediction."}, 500


# Data API
class GetData(Resource):
    def get(self):
        try:
            # Load the dataset
            data_path = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/Datasets/Mpesa_cleaned_data.csv"
            df = pd.read_csv(data_path)

            # Convert to JSON format
            res = df.to_json(orient="records")
            return res, 200  # Return the data with a 200 OK status

        except Exception as e:
            print(f"Error: {e}")
            return {"error": "An error occurred while fetching the data."}, 500


# Adding resources to the API
api.add_resource(Prediction, '/prediction')
api.add_resource(GetData, '/data')

if __name__ == "__main__":
    app.run(debug=True)
