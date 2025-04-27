from flask import Flask, request, render_template
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
import xgboost

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html")


CORS(app)

api = Api(app)


# Prediction API call
class Prediction(Resource):
    def get(self, transaction_amount):
        try:
            print(f"Received transaction amount: {transaction_amount}")

            # Create a DataFrame with 12 features (you can adjust this based on your training features)
            transaction_amount = [int(transaction_amount)]  # Convert to integer

            # Assuming your model was trained on 12 features, we create dummy columns for missing features
            # You may need to replace the dummy values with real values if available
            dummy_features = [0] * 11  # Assuming the model uses 11 other features
            features = transaction_amount + dummy_features  # Combine the transaction amount with the dummy features

            df = pd.DataFrame([features], columns=[f'feature_{i}' for i in range(1,
                                                                                 13)])  # Naming the columns as feature_1, feature_2, ..., feature_12

            # Load the trained model
            model_path = "/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/Mpesa_XGBRegressor.pkl"
            model = pickle.load(open(model_path, "rb"))

            # Make the prediction
            prediction = model.predict(df)
            prediction = int(prediction[0])  # Convert the prediction to integer

            return {"prediction": prediction}, 200  # Returning JSON response

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
