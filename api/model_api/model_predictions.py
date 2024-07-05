import pickle
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json
import math


load_dotenv() # Load Environment Variables 


# Input for model set as function's input
def machine_learning(x_input): 

    # Define Scalers
    scaler = StandardScaler() 
    poly = PolynomialFeatures(degree=2)
    # Define Empty List to store model name and predicted outcome
    model_list = []
    # Define path
    path = os.path.join("./model_api/","models/Supervised/")

    # Loop through all models and make predictions
    for model in os.listdir(path):
        if model.endswith('.pkl'):
            try:
                with open(os.path.join(path,model), 'rb') as f: # Unpickle the model
                    regressor = pickle.load(f)
                
                    if model == "Linear Regression.pkl": # No transformation for Linear Regressor
                        x_scaled = [x_input]
                    else: # Use StandardScaler for every other type of Regressor
            
                        x_scaled = scaler.fit_transform([x_input])
                    try:
                        prediction = regressor.predict(x_scaled)[0] # Predict and convert to float

                        if isinstance(prediction, np.ndarray):
                            prediction = prediction.item()  # Extract single element from ndarray if necessary
                    except:
                        continue

            except Exception as e:  # Catch exceptions if any
                print(e)
                continue

            model_list.append([model.split('.')[0], float(prediction)])  # Convert prediction to float and append to list

    return model_list

# Define the MLP regressor model
class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.hidden1 = nn.Linear(4, 64)  # input layer to first hidden layer
        self.hidden2 = nn.Linear(64, 32) # first hidden layer to second hidden layer
        self.output = nn.Linear(32, 1)   # second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Function to predict output using the MLP Regressor
def deep_learning(x_input):
    path = os.path.join("./model_api/","models/Neural Network")
    model_weights = os.listdir(path)
    model = MLPRegressor() # Create the Object
    model.load_state_dict(torch.load(os.path.join(path,model_weights[0]))) # Load the model weights
    model.eval() # Set the model to evaluation mode
    x_input = np.array(x_input, dtype=np.float32)

    if x_input.ndim == 1:
        x_input = x_input.reshape(1, -1)  # Reshape to (1, 4) if single sample

    if x_input.shape[1] != 4:
        raise ValueError("Input should have exactly four features")

    with torch.no_grad():
        prediction = model(torch.from_numpy(x_input)) # Make predictions
    return ["MLP Regressor", prediction.numpy()[0][0]]    

def single_predictions(x_input):
    response = machine_learning(x_input)
    response.append(deep_learning(x_input))
    response_json = [{"Model Name": item[0], "Predicted Score": float(item[1].item() if isinstance(item[1], np.ndarray) else item[1])} for item in response] #list with seperate dicts inside. Do response_json[index]['Model Name'] or ['Predicted Score'] to get the values
    # print(type(response_json))
    return response_json

# print(single_predictions(5))


def predict_with_model(df, input_columns, model_file, models_folder):
    scaler = StandardScaler()
    model_path = os.path.join(models_folder, model_file)
    
    with open(model_path, 'rb') as f:
        regressor = pickle.load(f)
        model_name = os.path.splitext(model_file)[0]

        if model_file != "Linear Regression.pkl":
            df[input_columns] = scaler.fit_transform(df[input_columns])
            
        predictions = regressor.predict(df[input_columns])
        predictions = [min(0.99, float(pred)) for pred in predictions]

    return model_name, predictions

def bulk_prediction(df):
    df = text_classify_transformer(df)
    input_columns = ['Authentication and Access Anomalies','Impersonation and Phishing','Threats and Malicious Activity','File and Search Activity']
    models_folder = "./model_api/models/Supervised/"

    def safe_predict(model_file):
        try:
            return predict_with_model(df, input_columns, model_file, models_folder)
        except Exception as e:
            print(f"Error with model {model_file}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
        results = list(executor.map(safe_predict, model_files))

    results = [result for result in results if result]
    for model_name, predictions in results:
        df[f'{model_name} Predictions'] = predictions

    return df

def bulk_deep_learning(df):
    input_columns = ['Authentication and Access Anomalies','Impersonation and Phishing','Threats and Malicious Activity','File and Search Activity']
    path = os.path.join("./model_api/","models/Neural Network")
    model_weights = os.listdir(path)
    model = MLPRegressor() # Create the Object
    model.load_state_dict(torch.load(os.path.join(path,model_weights[0]))) # Load the model weights
    model.eval() # Set the model to evaluation mode

    # Ensure x_input is a 2D tensor
    x_input = np.array(df[input_columns], dtype=np.float32)

    if x_input.ndim == 1:
        x_input = x_input.reshape(1, -1)  # Reshape to (1, 4) if single sample

    if x_input.shape[1] != 4:
        raise ValueError("Input should have exactly four features")

    with torch.no_grad():
        predictions = model(torch.from_numpy(x_input)) # Make predictions
        # Convert the tensor to a numpy array
        predictions = predictions.numpy()
    
        # Flatten the predictions
        predictions = predictions.flatten().tolist()    
        predictions = [min(0.99, pred) for pred in predictions]
    if predictions is not None:
            df['MLP Regressor Predictions'] = predictions
    
    return df

def text_classify_transformer(df):
    newdf = df.groupby('UPN').agg(    
        UPN_Count=('UPN', 'size'),
        Authentication_and_Access_Anomalies=('Classified Description', lambda x: (x == 'Authentication and Access Anomalies').sum()),
        Impersonation_and_Phishing=('Classified Description', lambda x: (x == 'Impersonation and Phishing').sum()),
        Threats_and_Malicious_Activity=('Classified Description', lambda x: (x == 'Threats and Malicious Activity').sum()),
        File_and_Search_Activity=('Classified Description', lambda x: (x == 'File and Search Activity').sum())
    ).reset_index()

    newdf.rename(columns={
        'Authentication_and_Access_Anomalies': 'Authentication and Access Anomalies', 
        'Impersonation_and_Phishing': 'Impersonation and Phishing',
        'Threats_and_Malicious_Activity': 'Threats and Malicious Activity',
        'File_and_Search_Activity': 'File and Search Activity'}, inplace=True)
    
    return newdf

def adjust_df(df):
    # Function to adjust predictions for a given column
    def adjust_predictions(df, prediction_column):
        # Sort by the specified prediction column and then by 'UPN_Count'
        df = df.sort_values(by=[prediction_column, 'UPN_Count'], ascending=[False, False])

        # Group by the specified prediction column and rank within the group
        def rank_within_group(group):
            group['Rank'] = group['UPN_Count'].rank(method='dense', ascending=True)
            return group

        df = df.groupby(prediction_column, group_keys=False).apply(rank_within_group)

        # Function to determine the adjustment factor
        def calculate_adjustment_factor(rank):
            n = len(str(int(rank)))
            return 0.001 * math.pow(10, 1-n)

        df['Max_Rank'] = df.groupby(prediction_column)['Rank'].transform('max')

        # Determine the adjustment factor for each group
        df['Adjustment_Factor'] = df['Max_Rank'].apply(calculate_adjustment_factor)

        # Apply the adjustment factor dynamically
        df['Adjustment'] = df['Rank'] * df['Adjustment_Factor']

        # Adjust the specified prediction column
        df[prediction_column] += df['Adjustment']

        # Drop unnecessary columns
        df = df.drop(columns=['Rank', 'Max_Rank', 'Adjustment_Factor', 'Adjustment'])

        return df

    # Adjust predictions for both 'MLP Regressor Predictions' and 'Linear Regression Predictions'
    df = adjust_predictions(df, 'MLP Regressor Predictions')
    df = adjust_predictions(df, 'Linear Regression Predictions')

    # Reset index
    df = df.reset_index(drop=True)

    return df