# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import mysql.connector
# import time
# import os

# if not hasattr(time, 'clock'):
#     time.clock = time.perf_counter
    
# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open('model.pkl', 'rb'))

# # MySQL Database configuration
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="password",
#     database="fraud_detection_dbb"
# )

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         account_number = request.form['accountNumber']
#         customer_id = request.form['customerId']
#         credit_limit = float(request.form['creditLimit'])
#         available_money = float(request.form['availableMoney'])
#         transaction_amount = float(request.form['transactionAmount'])
#         current_balance = float(request.form['currentBalance'])
#         card_present = int(request.form['cardPresent'])
#         expiration_match = int(request.form['expirationDateKeyInMatch'])

#         # Prepare input data for prediction
#         input_data = np.array([[credit_limit, available_money, transaction_amount, current_balance, card_present, expiration_match]])

#         # Make the prediction
#         prediction = model.predict(input_data)
#         prediction_text = "Fraudulent" if prediction[0] == 1 else "Legitimate"

#         # Insert the form data and prediction result into the database
#         cursor = db.cursor()
#         query = """
#             INSERT INTO transactions (account_number, customer_id, credit_limit, available_money, transaction_amount,
#                                       current_balance, card_present, expiration_date_key_in_match, prediction)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """
#         values = (account_number, customer_id, credit_limit, available_money, transaction_amount,
#                   current_balance, card_present, expiration_match, prediction_text)
#         cursor.execute(query, values)
#         db.commit()
#         cursor.close()

#         return render_template('result.html', result=f"Prediction: {prediction_text}")

#     except Exception as e:
#         return f"Error: {str(e)}"


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))



#####################streamlite testing########################33333

# import streamlit as st
# import pandas as pd
# import joblib
# import gdown
# import os
# # Download model and dataset from Google Drive
# @st.cache_data
# def load_model():
#     url = "https://drive.google.com/file/d/1EUbs_Z75_P30Gxmr395kHv5VetBGzjtG/view?usp=drive_link"  # Replace with your model file ID
#     output = "model.pkl"
#     # gdown.download(url, output, quiet=False)
#     # model = joblib.load(output)
#     # return model
#     if not os.path.exists(output):  # Download only if not already downloaded
#         gdown.download(url, output, quiet=False)
#     model = joblib.load(output)
#     return model

# @st.cache_data
# def load_data():
#     url = "https://drive.google.com/file/d/1tlL8-z2ZzYDkRTRk-9KRRAIRCQFybDF6/view?usp=drive_link"  # Replace with your dataset file ID
#     output = "Data.csv"
#     if not os.path.exists(output):  # Download only if not already downloaded
#         gdown.download(url, output, quiet=False)
#     data = pd.read_csv(output)
#     return data
#     # gdown.download(url, output, quiet=False)
#     # data = pd.read_csv(output)
#     # return data

# # App UI
# st.title("Fraud Detection System")
# st.write("Upload transaction data to detect fraud.")

# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
# if uploaded_file:
#     input_data = pd.read_csv(uploaded_file)
#     model = load_model()
#     predictions = model.predict(input_data)
#     st.write("Predictions:")
#     st.write(predictions)
# else:
#     st.write("Using sample data.")
#     data = load_data()
#     st.write(data.head())


#####################when there is error#####################
###################testing the csv file so it is working down wala ########################

# import streamlit as st
# import pandas as pd

# # CSV file ko load karne ka function
# def load_csv(file):
#     try:
#         # File ko load karna
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')  # Delimiter ',' ya ';' check karein
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")  # Agar CSV file mein format error ho
#         return pd.DataFrame()  # Empty DataFrame return karein
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")  # Dusri errors ke liye
#         return pd.DataFrame()

# # Streamlit app ke UI elements
# st.title("Fraud Detection System")
# st.write("Upload apna transaction data fraud detect karne ke liye.")

# # User se file upload karwane ka option
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
# if uploaded_file:
#     st.write("Uploaded file detect ho gayi.")  # File upload success message
#     data = load_csv(uploaded_file)  # File ko process karo
#     if not data.empty:
#         st.write("File load ho gayi successfully!")  # File properly load ho gayi
#         st.write(data.head())  # Uploaded file ki first 5 rows dikhata hai
#     else:
#         st.write("File load nahi ho paayi. Format ya delimiter check karein.")
# else:
#     st.write("Please ek CSV file upload karein.")  # Jab tak file upload nahi hoti



#######################now testing finnalyy with frontend####################3
# import streamlit as st
# import pandas as pd
# import joblib

# # CSV file load karne ka function
# def load_csv(file):
#     try:
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")
#         return pd.DataFrame()

# # Model load karne ka function
# @st.cache_resource
# def load_model():
#     model_path = "model.pkl"  # Model file ka path
#     model = joblib.load(model_path)
#     return model

# # Prediction karne ka function
# def make_predictions(data, model):
#     try:
#         predictions = model.predict(data)
#         return predictions
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # Streamlit App ka UI
# st.title("Fraud Detection System")
# st.write("Upload apna transaction data fraud detect karne ke liye.")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file:
#     st.write("Uploaded file detect ho gayi.")
#     data = load_csv(uploaded_file)
#     if not data.empty:
#         st.write("File load ho gayi successfully!")
#         st.write("Data preview:")
#         st.write(data.head())

#         # Load the model
#         model = load_model()

#         # Predict on data
#         st.write("Predictions:")
#         predictions = make_predictions(data, model)
#         if predictions is not None:
#             st.write(predictions)
#     else:
#         st.write("File load nahi ho paayi. Format ya delimiter check karein.")
# else:
#     st.write("Please ek CSV file upload karein.")


####################newnote giving correct prediction ========================================

# import streamlit as st
# import pandas as pd
# import joblib

# # CSV file load karne ka function
# def load_csv(file):
#     try:
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")
#         return pd.DataFrame()

# # Model load karne ka function
# @st.cache_resource
# def load_model():
#     model_path = "model.pkl"  # Model file ka path
#     model = joblib.load(model_path)
#     return model

# # Data preprocessing function
# def preprocess_data(data):
#     # Feature selection
#     features = [
#         'creditLimit',
#         'availableMoney',
#         'transactionAmount',
#         'currentBalance',
#         'cardPresent',
#         'expirationDateKeyInMatch'
#     ]
    
#     # Drop unnecessary columns
#     columns_to_drop = ['transactionDateTime', 'merchantName', 'merchantCity', 'echoBuffer']
#     for col in columns_to_drop:
#         if col in data.columns:
#             data.drop(columns=col, inplace=True)

#     # Handle missing values
#     data = data[features].dropna()

#     # Convert categorical features to numeric (Yes -> 1, No -> 0)
#     if 'cardPresent' in data.columns:
#         data['cardPresent'] = data['cardPresent'].apply(lambda x: 1 if x == 'Yes' else 0)
#     if 'expirationDateKeyInMatch' in data.columns:
#         data['expirationDateKeyInMatch'] = data['expirationDateKeyInMatch'].apply(lambda x: 1 if x == 'Yes' else 0)

#     return data

# # Prediction karne ka function
# def make_predictions(data, model):
#     try:
#         predictions = model.predict(data)
#         return predictions
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # Streamlit App ka UI
# st.title("Fraud Detection System")
# st.write("Upload apna transaction data fraud detect karne ke liye.")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file:
#     st.write("Uploaded file detect ho gayi.")
#     data = load_csv(uploaded_file)
#     if not data.empty:
#         st.write("File load ho gayi successfully!")
#         st.write("Data preview:")
#         st.write(data.head())

#         # Preprocess the data
#         processed_data = preprocess_data(data)

#         # Check if processed data is not empty
#         if not processed_data.empty:
#             st.write("Preprocessed data preview:")
#             st.write(processed_data.head())

#             # Load the model
#             model = load_model()

#             # Predict on data
#             st.write("Predictions:")
#             predictions = make_predictions(processed_data, model)
#             if predictions is not None:
#                 st.write(predictions)
#         else:
#             st.write("Preprocessed data empty hai. Check karen ki correct format mein data diya ho.")
#     else:
#         st.write("File load nahi ho paayi. Format ya delimiter check karein.")
# else:
#     st.write("Please ek CSV file upload karein.")


######################33trying for proper result 
# import streamlit as st
# import pandas as pd
# import joblib

# # CSV file load karne ka function
# def load_csv(file):
#     try:
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")
#         return pd.DataFrame()

# # Model load karne ka function
# @st.cache_resource
# def load_model():
#     model_path = "model.pkl"  # Model file ka path
#     model = joblib.load(model_path)
#     return model

# # Data preprocessing function
# def preprocess_data(data):
#     # Feature selection (only the required 6 features for prediction)
#     features = [
#         'creditLimit',
#         'availableMoney',
#         'transactionAmount',
#         'currentBalance',
#         'cardPresent',
#         'expirationDateKeyInMatch'
#     ]
    
#     # Drop unnecessary columns (only keep the features for prediction)
#     columns_to_drop = ['transactionDateTime', 'merchantName', 'merchantCity', 'echoBuffer']
#     for col in columns_to_drop:
#         if col in data.columns:
#             data.drop(columns=col, inplace=True)

#     # Ensure only the required columns are passed to the model
#     data = data[features].dropna()

#     # Convert categorical features to numeric (Yes -> 1, No -> 0)
#     if 'cardPresent' in data.columns:
#         data['cardPresent'] = data['cardPresent'].apply(lambda x: 1 if x == 'Yes' else 0)
#     if 'expirationDateKeyInMatch' in data.columns:
#         data['expirationDateKeyInMatch'] = data['expirationDateKeyInMatch'].apply(lambda x: 1 if x == 'Yes' else 0)

#     return data

# # Prediction karne ka function
# def make_predictions(data, model):
#     try:
#         # Ensure data is in the correct shape (2D array)
#         data_2d = data.values  # No need to reshape if data is already 2D
#         predictions = model.predict(data_2d)
#         return predictions
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # Streamlit App ka UI
# st.title("Fraud Detection System")
# st.write("Upload apna transaction data fraud detect karne ke liye.")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file:
#     st.write("Uploaded file detect ho gayi.")
#     data = load_csv(uploaded_file)
#     if not data.empty:
#         st.write("File load ho gayi successfully!")
#         st.write("Data preview:")
#         st.write(data.head())

#         # Preprocess the data
#         processed_data = preprocess_data(data)

#         # Check if processed data is not empty
#         if not processed_data.empty:
#             st.write("Preprocessed data preview:")
#             st.write(processed_data.head())

#             # Load the model
#             model = load_model()

#             # Predict on data
#             st.write("Predictions (0: non-fraud, 1: fraud):")
#             predictions = make_predictions(processed_data, model)
#             if predictions is not None:
#                 st.write(predictions)
#         else:
#             st.write("Preprocessed data empty hai. Check karen ki correct format mein data diya ho.")
#     else:
#         st.write("File load nahi ho paayi. Format ya delimiter check karein.")
# else:
#     st.write("Please ek CSV file upload karein.")
###############################33upar wla still not correct pre##############33


##############33333now testing ##################

# import streamlit as st
# import pandas as pd
# import joblib

# # CSV file load karne ka function
# def load_csv(file):
#     try:
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")
#         return pd.DataFrame()

# # Model load karne ka function
# @st.cache_resource
# def load_model():
#     model_path = "model.pkl"  # Model file ka path
#     model = joblib.load(model_path)
#     return model

# # Data preprocessing function
# def preprocess_data(data):
#     # Feature selection (only the required 6 features for prediction)
#     features = [
#         'creditLimit',
#         'availableMoney',
#         'transactionAmount',
#         'currentBalance',
#         'cardPresent',
#         'expirationDateKeyInMatch'
#     ]
    
#     # Drop unnecessary columns (only keep the features for prediction)
#     columns_to_drop = ['transactionDateTime', 'merchantName', 'merchantCity', 'echoBuffer']
#     for col in columns_to_drop:
#         if col in data.columns:
#             data.drop(columns=col, inplace=True)

#     # Ensure only the required columns are passed to the model
#     data = data[features].dropna()

#     # Convert categorical features to numeric (Yes -> 1, No -> 0)
#     if 'cardPresent' in data.columns:
#         data['cardPresent'] = data['cardPresent'].apply(lambda x: 1 if x == 'Yes' else 0)
#     if 'expirationDateKeyInMatch' in data.columns:
#         data['expirationDateKeyInMatch'] = data['expirationDateKeyInMatch'].apply(lambda x: 1 if x == 'Yes' else 0)

#     return data

# # Prediction karne ka function
# def make_predictions(data, model):
#     try:
#         # Ensure data is in the correct shape (2D array)
#         data_2d = data.values  # No need to reshape if data is already 2D
#         predictions = model.predict(data_2d)
#         return predictions
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # Streamlit App ka UI
# st.title("Fraud Detection System")
# st.write("Upload apna transaction data fraud detect karne ke liye.")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file:
#     st.write("Uploaded file detect ho gayi.")
#     data = load_csv(uploaded_file)
#     if not data.empty:
#         st.write("File load ho gayi successfully!")
#         st.write("Data preview:")
#         st.write(data.head())

#         # Preprocess the data
#         processed_data = preprocess_data(data)

#         # Check the unique values in the columns
#         st.write("Unique values in 'cardPresent':", processed_data['cardPresent'].unique())
#         st.write("Unique values in 'expirationDateKeyInMatch':", processed_data['expirationDateKeyInMatch'].unique())

#         # Show the preprocessed data
#         st.write("Preprocessed data preview:")
#         st.write(processed_data.head())

#         # Load the model
#         model = load_model()

#         # Predict on data
#         st.write("Predictions (0: non-fraud, 1: fraud):")
#         predictions = make_predictions(processed_data, model)
#         if predictions is not None:
#             st.write(predictions)
#         else:
#             st.error("Prediction is empty or there was an issue with the model.")
#     else:
#         st.write("File load nahi ho paayi. Format ya delimiter check karein.")
# else:
#     st.write("Please ek CSV file upload karein.")







##########################3checking###################
# import streamlit as st
# import pandas as pd
# import joblib

# # CSV file load karne ka function
# def load_csv(file):
#     try:
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")
#         return pd.DataFrame()

# # Model load karne ka function
# @st.cache_resource
# def load_model():
#     model_path = "model.pkl"  # Model file ka path
#     model = joblib.load(model_path)
#     return model

# # Data preprocessing function
# def preprocess_data(data):
#     # Feature selection (only the required 6 features for prediction)
#     required_features = [
#         'creditLimit',
#         'availableMoney',
#         'transactionAmount',
#         'currentBalance',
#         'cardPresent',
#         'expirationDateKeyInMatch'
#     ]
    
#     # Drop unnecessary columns (only keep the features for prediction)
#     columns_to_drop = ['transactionDateTime', 'merchantName', 'merchantCity', 'echoBuffer']
#     for col in columns_to_drop:
#         if col in data.columns:
#             data.drop(columns=col, inplace=True)

#     # Ensure only the required columns are passed to the model
#     data = data[required_features].dropna()

#     # Convert categorical features to numeric (Yes -> 1, No -> 0)
#     if 'cardPresent' in data.columns:
#         data['cardPresent'] = data['cardPresent'].apply(lambda x: 1 if x == 'Yes' else 0)
#     if 'expirationDateKeyInMatch' in data.columns:
#         data['expirationDateKeyInMatch'] = data['expirationDateKeyInMatch'].apply(lambda x: 1 if x == 'Yes' else 0)

#     return data

# # Prediction karne ka function
# def make_predictions(data, model):
#     try:
#         # Ensure data is in the correct shape (2D array)
#         data_2d = data.values  # No need to reshape if data is already 2D
#         predictions = model.predict(data_2d)
#         return predictions
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # Streamlit App ka UI
# st.title("Fraud Detection System")
# st.write("Upload apna transaction data fraud detect karne ke liye.")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file:
#     st.write("Uploaded file detect ho gayi.")
#     data = load_csv(uploaded_file)
#     if not data.empty:
#         st.write("File load ho gayi successfully!")
#         st.write("Data preview:")
#         st.write(data.head())

#         # Preprocess the data
#         processed_data = preprocess_data(data)

#         # Check the unique values in the columns
#         st.write("Unique values in 'cardPresent':", processed_data['cardPresent'].unique())
#         st.write("Unique values in 'expirationDateKeyInMatch':", processed_data['expirationDateKeyInMatch'].unique())

#         # Show the preprocessed data
#         st.write("Preprocessed data preview:")
#         st.write(processed_data.head())

#         # Check for missing values
#         st.write("Missing values in processed data:", processed_data.isnull().sum())

#         # Show the shape of processed data
#         st.write("Processed data shape:", processed_data.shape)

#         # Load the model
#         model = load_model()

#         # Predict on data
#         st.write("Predictions (0: non-fraud, 1: fraud):")
#         predictions = make_predictions(processed_data, model)
#         if predictions is not None:
#             st.write(predictions)
#         else:
#             st.error("Prediction is empty or there was an issue with the model.")
#     else:
#         st.write("File load nahi ho paayi. Format ya delimiter check karein.")
# else:
#     st.write("Please ek CSV file upload karein.")


#####################33333trying again, inncorrect result =================================
# import streamlit as st
# import pandas as pd
# import joblib

# # **Function to load CSV file**
# def load_csv(file):
#     try:
#         data = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', delimiter=',')
#         st.write("Raw data preview (first 5 rows):")
#         st.write(data.head())

#         # Display unique values for debug
#         if 'cardPresent' in data.columns:
#             st.write("Unique values in raw 'cardPresent':", data['cardPresent'].unique())
#         else:
#             st.error("Column 'cardPresent' is missing in the uploaded file.")

#         if 'expirationDateKeyInMatch' in data.columns:
#             st.write("Unique values in raw 'expirationDateKeyInMatch':", data['expirationDateKeyInMatch'].unique())
#         else:
#             st.error("Column 'expirationDateKeyInMatch' is missing in the uploaded file.")
#         return data
#     except pd.errors.ParserError as e:
#         st.error(f"ParserError: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Koi aur error aayi: {e}")
#         return pd.DataFrame()

# # **Function to preprocess the data**
# def preprocess_data(data):
#     required_features = [
#         'creditLimit',
#         'availableMoney',
#         'transactionAmount',
#         'currentBalance',
#         'cardPresent',
#         'expirationDateKeyInMatch'
#     ]
#     columns_to_drop = ['transactionDateTime', 'merchantName', 'merchantCity', 'echoBuffer']

#     # Drop unnecessary columns
#     for col in columns_to_drop:
#         if col in data.columns:
#             data.drop(columns=col, inplace=True)

#     # Ensure required columns exist
#     missing_columns = [feature for feature in required_features if feature not in data.columns]
#     if missing_columns:
#         st.error(f"Missing required columns in uploaded data: {missing_columns}")
#         return pd.DataFrame()

#     # Handle cardPresent
#     if 'cardPresent' in data.columns:
#         data['cardPresent'] = data['cardPresent'].fillna('unknown').astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0)
#         st.write("Processed 'cardPresent' values:", data['cardPresent'].unique())

#     # Handle expirationDateKeyInMatch
#     if 'expirationDateKeyInMatch' in data.columns:
#         data['expirationDateKeyInMatch'] = data['expirationDateKeyInMatch'].fillna('unknown').astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0)
#         st.write("Processed 'expirationDateKeyInMatch' values:", data['expirationDateKeyInMatch'].unique())

#     # Filter to only the required features
#     data = data[required_features].dropna()

#     # Debug final processed data
#     st.write("Processed data preview:")
#     st.write(data.head())
#     st.write("Processed data shape:", data.shape)

#     return data

# # **Function to load the model**
# @st.cache_resource
# def load_model():
#     try:
#         model_path = "model.pkl"
#         model = joblib.load(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # **Function to make predictions**
# def make_predictions(data, model):
#     try:
#         predictions = model.predict(data)
#         return predictions
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # **Streamlit App**
# st.title("Fraud Detection System")
# st.write("Upload your transaction data in CSV format to detect fraudulent activity.")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file:
#     st.write("File detected successfully!")
#     data = load_csv(uploaded_file)

#     if not data.empty:
#         # Preprocess the uploaded data
#         processed_data = preprocess_data(data)

#         if not processed_data.empty:
#             # Load the model
#             model = load_model()

#             if model:
#                 # Make predictions
#                 predictions = make_predictions(processed_data, model)

#                 if predictions is not None:
#                     st.write("Fraud Detection Predictions (0: Non-Fraud, 1: Fraud):")
#                     st.write(predictions)
#         else:
#             st.write("Processed data is empty. Please ensure the uploaded file is valid.")
# else:
#     st.write("Please upload a CSV file.")




#########################33new start, working good but with streamlite frontend not mine one ##################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Error: Trained model file 'model.pkl' not found. Please check the file path.")
#         return None

# # Title of the Streamlit app
# st.title("Fraud Detection Prediction App")

# # Load the trained model
# model = load_model()

# if model:
#     # User input fields for prediction
#     st.sidebar.header("Input Features")
#     credit_limit = st.sidebar.number_input("Credit Limit", min_value=0.0, value=5000.0)
#     available_money = st.sidebar.number_input("Available Money", min_value=0.0, value=2000.0)
#     transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
#     current_balance = st.sidebar.number_input("Current Balance", min_value=0.0, value=1500.0)
#     card_present = st.sidebar.selectbox("Card Present", options=["Yes", "No"])
#     expiration_date_key_in_match = st.sidebar.selectbox("Expiration Date Key-in Match", options=["Yes", "No"])

#     # Convert categorical inputs to numeric
#     card_present = 1 if card_present == "Yes" else 0
#     expiration_date_key_in_match = 1 if expiration_date_key_in_match == "Yes" else 0

#     # Create a DataFrame for prediction
#     input_data = pd.DataFrame({
#         'creditLimit': [credit_limit],
#         'availableMoney': [available_money],
#         'transactionAmount': [transaction_amount],
#         'currentBalance': [current_balance],
#         'cardPresent': [card_present],
#         'expirationDateKeyInMatch': [expiration_date_key_in_match]
#     })

#     # Display input data
#     st.write("### Input Data for Prediction")
#     st.write(input_data)

#     # Predict using the model
#     if st.button("Predict"):
#         try:
#             prediction = model.predict(input_data)
#             result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
#             st.success(f"Prediction: {result}")
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")
# else:
#     st.error("Please ensure the model file and dependencies are correctly set up.")



#############################frontend adding trying######################

# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Trained model file 'model.pkl' not found.")
#         return None

# # Load the model
# model = load_model()

# # Streamlit app
# st.title("AI Fraud Detection System")

# # Input fields
# credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, step=0.01)
# available_money = st.number_input("Available Money", min_value=0.0, value=2000.0, step=0.01)
# transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
# current_balance = st.number_input("Current Balance", min_value=0.0, value=1500.0, step=0.01)
# card_present = st.selectbox("Card Present", ["Yes", "No"])
# expiration_date_key_in_match = st.selectbox("Expiration Date Key-In Match", ["Yes", "No"])

# # Convert categorical features to numeric
# card_present = 1 if card_present == "Yes" else 0
# expiration_date_key_in_match = 1 if expiration_date_key_in_match == "Yes" else 0

# # Prediction button
# if st.button("Predict"):
#     if model:
#         input_data = np.array([
#             credit_limit,
#             available_money,
#             transaction_amount,
#             current_balance,
#             card_present,
#             expiration_date_key_in_match
#         ]).reshape(1, -1)

#         prediction = model.predict(input_data)
#         result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"

#         # Display the result
#         st.success(f"Prediction: {result}")
#     else:
#         st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")
####################################working done , now adding the database==giving all legitmate #########################33

# import streamlit as st
# import mysql.connector
# import pickle
# import numpy as np
# from datetime import datetime

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Trained model file 'model.pkl' not found.")
#         return None

# # Function to connect to MySQL
# def connect_to_db():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",       # Replace with your MySQL host
#             user="root",       # Replace with your MySQL username
#             password="password",  # Replace with your MySQL password
#             database="FraudDetection"  # Replace with your database name
#         )
#         return conn
#     except mysql.connector.Error as e:
#         st.error(f"Error connecting to MySQL: {e}")
#         return None

# # Function to insert data into the database
# def insert_to_db(conn, data):
#     try:
#         cursor = conn.cursor()
#         sql_query = """
#         INSERT INTO Transactions (
#             credit_limit, available_money, transaction_amount, current_balance, 
#             card_present, expiration_date_key_in_match, prediction
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s);
#         """
#         cursor.execute(sql_query, data)
#         conn.commit()
#         cursor.close()
#         st.success("Data successfully saved to the database.")
#     except mysql.connector.Error as e:
#         st.error(f"Error inserting data: {e}")

# # Load the model
# model = load_model()

# # Streamlit app
# st.title("AI Fraud Detection System")

# # Input fields
# credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, step=0.01)
# available_money = st.number_input("Available Money", min_value=0.0, value=2000.0, step=0.01)
# transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
# current_balance = st.number_input("Current Balance", min_value=0.0, value=1500.0, step=0.01)
# card_present = st.selectbox("Card Present", ["Yes", "No"])
# expiration_date_key_in_match = st.selectbox("Expiration Date Key-In Match", ["Yes", "No"])

# # Convert categorical features to numeric
# card_present_numeric = 1 if card_present == "Yes" else 0
# expiration_date_key_in_match_numeric = 1 if expiration_date_key_in_match == "Yes" else 0

# # Prediction button
# if st.button("Predict"):
#     if model:
#         # Prepare input data for prediction
#         input_data = np.array([
#             credit_limit,
#             available_money,
#             transaction_amount,
#             current_balance,
#             card_present_numeric,
#             expiration_date_key_in_match_numeric
#         ]).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(input_data)
#         result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"

#         # Display the result
#         st.success(f"Prediction: {result}")

#         # Connect to MySQL and store the inputs and prediction
#         conn = connect_to_db()
#         if conn:
#             db_data = (
#                 credit_limit,
#                 available_money,
#                 transaction_amount,
#                 current_balance,
#                 card_present_numeric,
#                 expiration_date_key_in_match_numeric,
#                 result
#             )
#             insert_to_db(conn, db_data)
#             conn.close()
#     else:
#         st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")




#########################dont try now ######

# import pandas as pd
# import pickle

# # Load the trained model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Define the feature list for prediction (must match training features)
# features = [
#     'credit_limit', 
#     'available_money', 
#     'transaction_amount', 
#     'current_balance', 
#     'card_present_numeric', 
#     'expiration_date_key_in_match_numeric'
# ]

# # Prepare your input data (from frontend form or other sources)
# input_data = {
#     'credit_limit': [5000],
#     'available_money': [2000],
#     'transaction_amount': [100],
#     'current_balance': [1500],
#     'card_present_numeric': [1],  # Yes
#     'expiration_date_key_in_match_numeric': [1]  # Yes
# }

# # Convert input data to DataFrame
# input_df = pd.DataFrame(input_data)

# # Ensure the input data has the correct columns in the correct order
# input_df = input_df[features]

# # Predict using the model
# prediction = model.predict(input_df)

# # Output the prediction
# print(f"Prediction result: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")



#######################attractive frontend#####################3

# import streamlit as st
# import mysql.connector
# import pickle
# import numpy as np

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Trained model file 'model.pkl' not found.")
#         return None

# # Function to connect to MySQL
# def connect_to_db():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",       # Replace with your MySQL host
#             user="root",            # Replace with your MySQL username
#             password="password",    # Replace with your MySQL password
#             database="FraudDetection"  # Replace with your database name
#         )
#         return conn
#     except mysql.connector.Error as e:
#         st.error(f"Error connecting to MySQL: {e}")
#         return None

# # Function to insert data into the database
# def insert_to_db(conn, data):
#     try:
#         cursor = conn.cursor()
#         sql_query = """
#         INSERT INTO Transactions (
#             credit_limit, available_money, transaction_amount, current_balance, 
#             card_present, expiration_date_key_in_match, prediction
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s);
#         """
#         cursor.execute(sql_query, data)
#         conn.commit()
#         cursor.close()
#         st.success("Data successfully saved to the database.")
#     except mysql.connector.Error as e:
#         st.error(f"Error inserting data: {e}")

# # Load the model
# model = load_model()

# # Streamlit app
# st.markdown(
#     """
#     <style>
#         .main-title {
#             text-align: center;
#             font-size: 3rem;
#             font-weight: bold;
#             color: #2A9D8F;
#         }
#         .stButton button {
#             background-color: #264653;
#             color: white;
#             font-size: 1rem;
#             border-radius: 10px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown("<h1 class='main-title'>AI Fraud Detection System</h1>", unsafe_allow_html=True)
# st.markdown("---")

# # Create columns for better layout
# col1, col2 = st.columns(2)

# with col1:
#     credit_limit = st.number_input("💳 Credit Limit", min_value=0.0, value=5000.0, step=0.01)
#     transaction_amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0, step=0.01)
#     card_present = st.selectbox("📇 Card Present", ["Yes", "No"])

# with col2:
#     available_money = st.number_input("🏦 Available Money", min_value=0.0, value=2000.0, step=0.01)
#     current_balance = st.number_input("📊 Current Balance", min_value=0.0, value=1500.0, step=0.01)
#     expiration_date_key_in_match = st.selectbox("⏳ Expiration Date Key-In Match", ["Yes", "No"])

# # Convert categorical features to numeric
# card_present_numeric = 1 if card_present == "Yes" else 0
# expiration_date_key_in_match_numeric = 1 if expiration_date_key_in_match == "Yes" else 0

# # Add a predict button with a clean design
# st.markdown("---")
# if st.button("🔍 Predict"):
#     if model:
#         # Prepare input data for prediction
#         input_data = np.array([
#             credit_limit,
#             available_money,
#             transaction_amount,
#             current_balance,
#             card_present_numeric,
#             expiration_date_key_in_match_numeric
#         ]).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(input_data)
#         result = "🛑 Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"

#         # Display the result
#         st.markdown(
#             f"<h2 style='text-align:center; color:#264653;'>{result}</h2>",
#             unsafe_allow_html=True
#         )

#         # Connect to MySQL and store the inputs and prediction
#         conn = connect_to_db()
#         if conn:
#             db_data = (
#                 credit_limit,
#                 available_money,
#                 transaction_amount,
#                 current_balance,
#                 card_present_numeric,
#                 expiration_date_key_in_match_numeric,
#                 result
#             )
#             insert_to_db(conn, db_data)
#             conn.close()
#     else:
#         st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")


##########################33trying 10 nput wala #####################3
# import streamlit as st
# import mysql.connector
# import pickle
# import numpy as np

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Trained model file 'model.pkl' not found.")
#         return None

# # Function to connect to MySQL
# def connect_to_db():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",       # Replace with your MySQL host
#             user="root",            # Replace with your MySQL username
#             password="password",    # Replace with your MySQL password
#             database="FraudDetection"  # Replace with your database name
#         )
#         return conn
#     except mysql.connector.Error as e:
#         st.error(f"Error connecting to MySQL: {e}")
#         return None

# # Function to insert data into the database
# # Function to insert data into the database
# def insert_to_db(conn, data):
#     try:
#         cursor = conn.cursor()
#         sql_query = """
#         INSERT INTO Transactions (
#             credit_limit, available_money, transaction_amount, current_balance, 
#             card_present, expiration_date_key_in_match, money_left, 
#             current_balance_percentage, credit_usage_ratio, transaction_to_balance_ratio, prediction
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
#         """
#         cursor.execute(sql_query, data)
#         conn.commit()
#         cursor.close()
#         st.success("Data successfully saved to the database.")
#     except mysql.connector.Error as e:
#         st.error(f"Error inserting data: {e}")


# # Load the model
# model = load_model()

# # Streamlit app
# st.title("AI Fraud Detection System")
# st.markdown("---")

# # Create input fields
# col1, col2 = st.columns(2)

# with col1:
#     credit_limit = st.number_input("💳 Credit Limit", min_value=0.0, value=5000.0, step=0.01)
#     transaction_amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0, step=0.01)
#     available_money = st.number_input("🏦 Available Money", min_value=0.0, value=2000.0, step=0.01)
#     card_present = st.selectbox("📇 Card Present", ["Yes", "No"])

# with col2:
#     current_balance = st.number_input("📊 Current Balance", min_value=0.0, value=1500.0, step=0.01)
#     expiration_date_key_in_match = st.selectbox("⏳ Expiration Date Key-In Match", ["Yes", "No"])
#     money_left = st.number_input("💵 Money Left (Available Money - Transaction Amount)", min_value=0.0, value=1900.0, step=0.01)
#     current_balance_percentage = st.number_input("📈 Current Balance Percentage (Current Balance / Credit Limit)", min_value=0.0, value=0.3, step=0.01)

# # Additional calculated features
# credit_usage_ratio = st.number_input("📊 Credit Usage Ratio (Transaction Amount / Credit Limit)", min_value=0.0, value=0.02, step=0.01)
# transaction_to_balance_ratio = st.number_input("📉 Transaction to Balance Ratio (Transaction Amount / Current Balance)", min_value=0.0, value=0.07, step=0.01)

# # Convert categorical features to numeric
# card_present_numeric = 1 if card_present == "Yes" else 0
# expiration_date_key_in_match_numeric = 1 if expiration_date_key_in_match == "Yes" else 0

# # Predict button
# st.markdown("---")

# if st.button("🔍 Predict"):
#     if model:
#         # Prepare input data
#         input_data = np.array([
#             credit_limit,
#             available_money,
#             transaction_amount,
#             current_balance,
#             card_present_numeric,
#             expiration_date_key_in_match_numeric,
#             money_left,
#             current_balance_percentage,
#             credit_usage_ratio,
#             transaction_to_balance_ratio
#         ]).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(input_data)
#         result = "🛑 Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"

#         # Display result
#         st.success(f"Prediction: {result}")

#         # Save to database
#         conn = connect_to_db()
#         if conn:
#             db_data = (
#                 credit_limit, available_money, transaction_amount, current_balance,
#                 card_present_numeric, expiration_date_key_in_match_numeric, 
#                 money_left, current_balance_percentage, credit_usage_ratio, transaction_to_balance_ratio, 
#                 result
#             )
#             insert_to_db(conn, db_data)
#             conn.close()
#     else:
#         st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")



######################more attractive ######################33
# import streamlit as st
# import mysql.connector
# import pickle
# import numpy as np

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Trained model file 'model.pkl' not found.")
#         return None

# # Function to connect to MySQL
# def connect_to_db():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",       # Replace with your MySQL host
#             user="root",            # Replace with your MySQL username
#             password="password",    # Replace with your MySQL password
#             database="FraudDetection"  # Replace with your database name
#         )
#         return conn
#     except mysql.connector.Error as e:
#         st.error(f"Error connecting to MySQL: {e}")
#         return None

# # Function to insert data into the database
# def insert_to_db(conn, data):
#     try:
#         cursor = conn.cursor()
#         sql_query = """
#         INSERT INTO Transactions (
#             credit_limit, available_money, transaction_amount, current_balance, 
#             card_present, expiration_date_key_in_match, money_left, 
#             current_balance_percentage, credit_usage_ratio, transaction_to_balance_ratio, prediction
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
#         """
#         cursor.execute(sql_query, data)
#         conn.commit()
#         cursor.close()
#         st.success("Data successfully saved to the database.")
#     except mysql.connector.Error as e:
#         st.error(f"Error inserting data: {e}")

# # Load the model
# model = load_model()

# # Add custom CSS for styling
# st.markdown(
#     """
#     <style>
#     body {
#         background-image: url('https://www.transparenttextures.com/patterns/blueprint.png'); /* Background image */
#         background-size: cover;
#         color: #004d66; /* Text color */
#     }
#     .stButton>button {
#         background-color: #007acc; /* Button color */
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 10px;
#     }
#     .stButton>button:hover {
#         background-color: #005f99; /* Hover color */
#     }
#     h1, h3 {
#         color: #004d66;
#         text-align: center;
#     }
#     .stNumberInput, .stSelectbox {
#         background-color: #e6f7ff;
#         border-radius: 5px;
#         padding: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Streamlit app
# st.title("🌟 AI Fraud Detection System 🌟")
# st.markdown("---")

# # Create input fields
# st.markdown("<h3>🔍 Enter Transaction Details Below:</h3>", unsafe_allow_html=True)
# col1, col2 = st.columns(2)

# with col1:
#     credit_limit = st.number_input("💳 Credit Limit", min_value=0.0, value=5000.0, step=0.01)
#     transaction_amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0, step=0.01)
#     available_money = st.number_input("🏦 Available Money", min_value=0.0, value=2000.0, step=0.01)
#     card_present = st.selectbox("📇 Card Present", ["Yes", "No"])

# with col2:
#     current_balance = st.number_input("📊 Current Balance", min_value=0.0, value=1500.0, step=0.01)
#     expiration_date_key_in_match = st.selectbox("⏳ Expiration Date Key-In Match", ["Yes", "No"])
#     money_left = st.number_input("💵 Money Left (Available Money - Transaction Amount)", min_value=0.0, value=1900.0, step=0.01)
#     current_balance_percentage = st.number_input("📈 Current Balance Percentage (Current Balance / Credit Limit)", min_value=0.0, value=0.3, step=0.01)

# # Additional calculated features
# credit_usage_ratio = st.number_input("📊 Credit Usage Ratio (Transaction Amount / Credit Limit)", min_value=0.0, value=0.02, step=0.01)
# transaction_to_balance_ratio = st.number_input("📉 Transaction to Balance Ratio (Transaction Amount / Current Balance)", min_value=0.0, value=0.07, step=0.01)

# # Convert categorical features to numeric
# card_present_numeric = 1 if card_present == "Yes" else 0
# expiration_date_key_in_match_numeric = 1 if expiration_date_key_in_match == "Yes" else 0

# # Predict button
# st.markdown("---")

# if st.button("🔍 Predict"):
#     if model:
#         # Prepare input data
#         input_data = np.array([[
#             credit_limit, available_money, transaction_amount, current_balance,
#             card_present_numeric, expiration_date_key_in_match_numeric,
#             money_left, current_balance_percentage, credit_usage_ratio, transaction_to_balance_ratio
#         ]])

#         # Make prediction
#         prediction = model.predict(input_data)
#         result = "🛑 Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"

#         # Display result
#         st.success(f"Prediction: {result}")

#         # Save to database
#         conn = connect_to_db()
#         if conn:
#             db_data = (
#                 credit_limit, available_money, transaction_amount, current_balance,
#                 card_present_numeric, expiration_date_key_in_match_numeric,
#                 money_left, current_balance_percentage, credit_usage_ratio, transaction_to_balance_ratio,
#                 result
#             )
#             insert_to_db(conn, db_data)
#             conn.close()
#     else:
#         st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")






####################AS PER 24K DATA ===all done excellend , just adding frontend only ####################

# import streamlit as st
# import mysql.connector
# import pickle
# import numpy as np
# import pandas as pd  # Import pandas to handle the DataFrame

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open('model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         st.error("Trained model file 'model.pkl' not found.")
#         return None

# # Function to connect to MySQL
# def connect_to_db():
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",  # Replace with your MySQL host
#             user="root",       # Replace with your MySQL username
#             password="password",  # Replace with your MySQL password
#             database="FraudDetectionn"  # Replace with your database name
#         )
#         return conn
#     except mysql.connector.Error as e:
#         st.error(f"Error connecting to MySQL: {e}")
#         return None

# # Function to insert data into the database
# def insert_to_db(conn, data):
#     try:
#         cursor = conn.cursor()
#         sql_query = """
#         INSERT INTO Transactionss (
#             account_number, customer_id, credit_limit, available_money, 
#             transaction_amount, current_balance, card_present, expiration_date_key_in_match, prediction
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
#         """
#         cursor.execute(sql_query, data)
#         conn.commit()
#         cursor.close()
#         st.success("Data successfully saved to the database.")
#     except mysql.connector.Error as e:
#         st.error(f"Error inserting data: {e}")

# # Load the model
# model = load_model()

# # Streamlit app
# st.title("AI Fraud Detection System")
# st.markdown("---")

# # Create input fields
# account_number = st.text_input("Account Number")
# customer_id = st.text_input("Customer ID")
# credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, step=0.01)
# available_money = st.number_input("Available Money", min_value=0.0, value=2000.0, step=0.01)
# transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
# current_balance = st.number_input("Current Balance", min_value=0.0, value=1500.0, step=0.01)
# card_present = st.selectbox("Card Present", ["Yes", "No"])
# expiration_date_key_in_match = st.selectbox("Expiration Date Key-In Match", ["Yes", "No"])

# # Feature names used during model training (in camel case)
# feature_names = [
#     "creditLimit", "availableMoney", "transactionAmount", "currentBalance",
#     "cardPresent", "expirationDateKeyInMatch"
# ]

# # Predict button
# if st.button("Predict"):
#     if model:
#         # Convert categorical features to numeric
#         card_present_numeric = 1 if card_present == "Yes" else 0
#         expiration_date_key_in_match_numeric = 1 if expiration_date_key_in_match == "Yes" else 0

#         # Prepare input data as a Pandas DataFrame with feature names matching the model
#         input_data = pd.DataFrame([[credit_limit, available_money, transaction_amount, current_balance,
#                                      card_present_numeric, expiration_date_key_in_match_numeric]], columns=feature_names)

#         # Make prediction
#         prediction = model.predict(input_data)
#         result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"

#         # Display result
#         st.success(f"Prediction: {result}")

#         # Save to database
#         conn = connect_to_db()
#         if conn:
#             db_data = (
#                 account_number, customer_id, credit_limit, available_money, transaction_amount,
#                 current_balance, card_present_numeric, expiration_date_key_in_match_numeric, result
#             )
#             insert_to_db(conn, db_data)
#             conn.close()
#     else:
#         st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")



#########################with good frontend###################3
import streamlit as st
import mysql.connector
import pickle
import numpy as np
import pandas as pd  # Import pandas for DataFrame

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Trained model file 'model.pkl' not found.")
        return None

# Function to connect to MySQL
def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",  # Replace with your MySQL host
            user="root",       # Replace with your MySQL username
            password="password",  # Replace with your MySQL password
            database="FraudDetectionn"  # Replace with your database name
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Function to insert data into the database
def insert_to_db(conn, data):
    try:
        cursor = conn.cursor()
        sql_query = """
        INSERT INTO Transactionss (
            account_number, customer_id, credit_limit, available_money, 
            transaction_amount, current_balance, card_present, expiration_date_key_in_match, prediction
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(sql_query, data)
        conn.commit()
        cursor.close()
        st.success("Data successfully saved to the database.")
    except mysql.connector.Error as e:
        st.error(f"Error inserting data: {e}")

# Load the model
model = load_model()

# Streamlit app styling and structure
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .container {
            background-color: #f4f4f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .input-label {
            font-size: 14px;
            color: #4CAF50;
        }
        .input-box {
            width: 100%;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
            background-color: #fff;
        }
        .predict-btn {
            width: 100%;
            padding: 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .predict-btn:hover {
            background-color: #45a049;
        }
        .result {
            font-size: 20px;
            text-align: center;
            margin-top: 30px;
            color: #4CAF50;
        }
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 14px;
            color: #aaa;
            background-color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app with an icon
st.markdown('<h1 class="title">AI Fraud Detection System 🔒</h1>', unsafe_allow_html=True)

st.markdown("---")

# Create input fields
col1, col2 = st.columns([1, 1])

with col1:
    account_number = st.text_input("Account Number", key="account_number", placeholder="Enter account number")
    customer_id = st.text_input("Customer ID", key="customer_id", placeholder="Enter customer ID")
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, step=0.01)
    available_money = st.number_input("Available Money", min_value=0.0, value=2000.0, step=0.01)

with col2:
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
    current_balance = st.number_input("Current Balance", min_value=0.0, value=1500.0, step=0.01)
    card_present = st.selectbox("Card Present", ["Yes", "No"])
    expiration_date_key_in_match = st.selectbox("Expiration Date Key-In Match", ["Yes", "No"])

# Predict button with custom style
if st.button("Predict"):
    if model:
        # Convert categorical features to numeric (keeping camelCase names)
        card_present_numeric = 1 if card_present == "Yes" else 0
        expiration_date_key_in_match_numeric = 1 if expiration_date_key_in_match == "Yes" else 0

        # Prepare input data with camelCase names
        input_data = np.array([[credit_limit, available_money, transaction_amount, current_balance,
                                card_present_numeric, expiration_date_key_in_match_numeric]])

        # Create DataFrame with the correct column names (camelCase)
        feature_names = ['creditLimit', 'availableMoney', 'transactionAmount', 'currentBalance', 
                         'cardPresent', 'expirationDateKeyInMatch']  # Feature names in camelCase as used during training
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)
        result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"

        # Display result with an icon
        st.markdown(f'<div class="result">{result}</div>', unsafe_allow_html=True)

        # Save to database
        conn = connect_to_db()
        if conn:
            db_data = (
                account_number, customer_id, credit_limit, available_money, transaction_amount,
                current_balance, card_present_numeric, expiration_date_key_in_match_numeric, result
            )
            insert_to_db(conn, db_data)
            conn.close()
    else:
        st.error("Model not loaded. Please ensure 'model.pkl' is in the app directory.")

# Footer Section
st.markdown('<div class="footer">Fraud Detection System | Designed By Sheetal ❤️</div>', unsafe_allow_html=True)
