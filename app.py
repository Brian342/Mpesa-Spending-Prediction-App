from flask import Flask, request, render_template
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
# from sklearn.preprocessing import LabelEncoder, StandardScaler
from util import encode_categorical_columns, encode_categorical_columns_training_encoder

model_path = "Models/Mpesa_XGBRegressor.pkl"
scaling_type = "Models/Scaler.pkl"
encoding_type = "Models/df_encoding.pkl"

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
                'Transaction_type': transaction_type,
                'Transaction_party': transaction_party,
                'Transaction_amount': transaction_amount,
                'paid_in_or_Withdraw': paid_in_or_withdrawal,
                'Balance': 1000
            }

            input_df = pd.DataFrame([features])

            encode_input_df = encode_categorical_columns(input_df, encode)

            # Scale entire dataframe
            Scaled_input = scaler.transform(encode_input_df)

            predicted_spending = model.predict(Scaled_input)
            predicted_spending = int(predicted_spending[0])

            return {"PredictedSpending": predicted_spending}, 200  # Returning JSON response

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
