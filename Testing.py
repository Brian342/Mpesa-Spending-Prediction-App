import os
file_path = 'Models/Mpesa_LinearRegression.pkl'
try:
    with open(file_path, 'rb') as f:
        print(f"[✔️]Managed to open file: {os.path.basename(file_path)}")
except FileNotFoundError:
    print(f"[𝒳]Could not open the file {file_path}")
except Exception as e:
    print(f"[𝑥]An error occurred when opening the file {e}")
