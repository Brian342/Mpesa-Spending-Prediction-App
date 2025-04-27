from flask import Flask, request, render_template
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
import xgboost

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index2.html")


CORS(app)

api = Api(app)


# Prediction API call
class Prediction(Resource):
    def get(self, transaction_amount):
        try:
            print(f"Received transaction amount: {transaction_amount}")

            transaction_amount = [int(transaction_amount)]  # Convert to integer

            features = transaction_amount

            df = pd.DataFrame([features], columns=['transaction_amount'])  # Naming the columns as feature_1, feature_2, ..., feature_12

            # Load the trained model
            model_path = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/Mpesa_XGBRegressor_balance.pkl"
            model = pickle.load(open(model_path, "rb"))

            # Make the prediction
            prediction = model.predict(df)
            balance_after = int(prediction[0])

            return {"prediction": balance_after}, 200  # Returning JSON response

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
api.add_resource(Prediction, '/prediction/<int:transaction_amount>')
api.add_resource(GetData, '/data')

if __name__ == "__main__":
    app.run(debug=True)
