from flask import Flask, request, render_template
from flask_restful import Resource, Api
import pickle
import pandas as pd
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

            dummy_features = [0] * 11
            features = Transaction_amount + dummy_features

            df = pd.DataFrame([features], columns=[f'feature_{i}' for i in range(1, 13)])

            df_encoding = encode.transform(df[['Transaction_party']])
            df_encoding = pd.DataFrame(df_encoding, columns=['Transaction_party'])

            df_scaling = scaler.transform(df.drop(['Transaction_party']))
            df_encoded_scaling = pd.concat([df_encoding, pd.DataFrame(df_scaling)], axis=1)

            #  Make the prediction
            # prediction = model.predict(df)
            # prediction = int(prediction[0])

            predicted_spending = model.predict(df_encoded_scaling)
            predicted_spending = int(predicted_spending[0])

            starting_balance = 5000
            remaining_balance = starting_balance - predicted_spending

            return {"remaining_balance": remaining_balance}, 200  # Returning JSON response

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
