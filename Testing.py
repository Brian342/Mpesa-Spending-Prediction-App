import pickle

try:
    with open('/Users/briankimanzi/Documents/programming Languages/PythonProgramming/JupyterNoteBook/ModelsPrediction/df_encoding.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
