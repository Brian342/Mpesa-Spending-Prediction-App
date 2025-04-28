from flask import Flask, request, render_template
from flask_restful import Resource, Api
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index2.html")


CORS(app)

api = Api(app)

model_path = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/Mpesa_LinearRegression.pkl"
scaling_type = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/Scaler.pkl"
encoding_type = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/df_encoding.pkl"

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaling_type, "rb"))
encode = pickle.load(open(encoding_type, "rb"))


# Prediction API call
class Prediction(Resource):
    def get(self, Transaction_amount):
        try:
            print(f"Received transaction amount: {Transaction_amount}")

            Transaction_amount = [int(Transaction_amount)]

# features

            dummy_features = [
                np.random.randint(1, 32),  # transaction_Day (1-31)
                np.random.randint(2000, 2030),  # Year
                np.random.randint(1, 13),  # Month
                np.random.randint(1, 32),  # Date
                np.random.randint(0, 7),  # Weekday (0=Monday, 6=Sunday)
                np.random.randint(0, 24),  # Hour
                np.random.randint(0, 60),  # Minute
                np.random.randint(0, 60),  # Seconds
                np.random.choice(['SEND MONEY', 'M-SHWARI DEPOSIT FROM M-PESA', 'RECEIVED FUNDS',
                                  'PAY BILL', 'BUY GOODS', 'POCHI LA BIASHARA',
                                  'M-SHWARI WITHDRAW FROM M-PESA', 'PAY BILL CHARGES',
                                  'AIRTIME PURCHASE', 'CASH WITHDRAWAL CHARGES',
                                  'CASH WITHDRAWAL', 'BUSINESS PAYMENT', 'AGENT DEPOSIT', ]),  # Transaction_type
                np.random.choice(['Safaricom', 'Equity', 'Cooperative', 'KCB']),  # Transaction_party
                Transaction_amount,  # Transaction_amount (your real input)
                np.random.choice(['paid_in', 'withdrawn']),
                np.random.randint(0, 100000)
            ]
            columns = [
                'transaction_Day', 'Year', 'Month', 'Date', 'Weekday', 'Hour', 'Minute',
                'Seconds', 'Transaction_type', 'Transaction_party',
                'Transaction_amount', 'paid_in_or_Withdraw', 'Balance'
            ]

            df = pd.DataFrame([dummy_features], columns=columns)

            # if 'Transaction_party' in df.columns:
            #     df_encoding = encode.transform(df[['Transaction_party']])
            #     df_encoding = pd.DataFrame(df_encoding, columns=['Transaction_party'])
            # else:
            #     df_encoding = pd.DataFrame({'Transaction_party': ['Unknown']})  # default placeholder
            #
            # df_scaling = scaler.transform(df.drop(['Transaction_party'], axis=1, errors='ignore'))
            # df_encoded_scaling = pd.concat([df_encoding, pd.DataFrame(df_scaling)], axis=1)
            # Encode categorical columns

            for col in ['Transaction_type', 'Transaction_party', 'paid_in_or_Withdraw']:
                if col in df.columns:
                    df[col] = encode.transform(df[[col]])

            # Scale entire dataframe
            df_scaled = scaler.transform(df)

            # Predict
            predicted_spending = model.predict(df_scaled)
            predicted_spending = int(predicted_spending[0])

            #  Make the prediction
            # prediction = model.predict(df)
            # prediction = int(prediction[0])

            predicted_spending = model.predict(df_scaled)
            predicted_spending = int(predicted_spending[0])

            # starting_balance = 5000
            # remaining_balance = starting_balance - predicted_spending

            return {"Predicting_spending": predicted_spending}, 200  # Returning JSON response

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
api.add_resource(Prediction, '/prediction/<int:Transaction_amount>')
api.add_resource(GetData, '/data')

if __name__ == "__main__":
    app.run(debug=True)
