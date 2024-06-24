import pickle
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json


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
            with open(os.path.join(path,model), 'rb') as f: # Unpickle the model
                regressor = pickle.load(f)
            try:
                if model == "Linear Regression.pkl": # No transformation for Linear Regressor
                    x_scaled = [[x_input]]
                else: # Use StandardScaler for every other type of Regressor
                    x_scaled = scaler.fit_transform([[x_input]])

                prediction = regressor.predict(x_scaled)[0] # Predict and convert to float

                if isinstance(prediction, np.ndarray):
                    prediction = prediction.item()  # Extract single element from ndarray if necessary

            except Exception as e:  # Catch exceptions if any
                print(e)
                continue

            model_list.append([model.split('.')[0], float(prediction)])  # Convert prediction to float and append to list

    return model_list

# Define the MLP regressor model
class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.hidden1 = nn.Linear(1, 64)  # input layer to first hidden layer
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

    # Ensure x_input is a 2D tensor
    x_input = np.array(x_input, dtype=np.float32)
    if x_input.ndim == 0:
        x_input = x_input.reshape(1, 1)
    elif x_input.ndim == 1:
        x_input = x_input.reshape(-1, 1)

    with torch.no_grad():
        prediction = model(torch.from_numpy(x_input)) # Make predictions
    return ["MLP Regressor", np.array(prediction.numpy()[0][0])]
    

def single_predictions(x_input):
    response = machine_learning(x_input)
    response.append(deep_learning(x_input))
    response_json = [{"Model Name": item[0], "Predicted Score": float(item[1].item() if isinstance(item[1], np.ndarray) else item[1])} for item in response] #list with seperate dicts inside. Do response_json[index]['Model Name'] or ['Predicted Score'] to get the values
    # print(type(response_json))
    return response_json

# print(single_predictions(5))


def predict_with_model(df, input_column, model_file, models_folder):
    # Define Scalers
    scaler = StandardScaler()
    
    if model_file.endswith('.pkl'):
        model_path = os.path.join(models_folder, model_file)
        with open(model_path, 'rb') as f:  # Unpickle the model
            regressor = pickle.load(f)
        
        # Extract the model name (without file extension) to use in the column name
        model_name = os.path.splitext(model_file)[0]

        try:
            if model_file == "Linear Regression.pkl":  # No transformation for Linear Regressor
                x_scaled = df[[input_column]]
            else:  # Use StandardScaler for every other type of Regressor
                x_scaled = scaler.fit_transform(df[[input_column]])

            predictions = regressor.predict(x_scaled)  # Predict
            # Check and cap predictions at 1
            predictions = [min(0.9999999999999999, pred) for pred in predictions]

        except Exception as e:  # Catch exceptions if any
            print(f"Error processing model {model_name}: {e}")
            predictions = None

        return model_name, predictions
    
def bulk_prediction(df):

    # Specify the column to be used for predictions
    input_column = 'UPN Count'

    # Path to the folder containing the ML models
    models_folder = os.path.join("./model_api/","models/Supervised/")

    # Use ThreadPoolExecutor to parallelize model loading and predictions
    with ThreadPoolExecutor() as executor:
        # List of all .pkl files in the models folder
        model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
        
        # Execute predictions in parallel
        results = list(executor.map(lambda model_file: predict_with_model(df, input_column, model_file, models_folder), model_files))

    # Add predictions to the dataframe
    for model_name, predictions in results:
        if predictions is not None:
            df[f'{model_name} Predictions'] = predictions

    # Save the updated dataframe to a new CSV file
    return df

def bulk_deep_learning(df):
    input_column = 'UPN Count'
    path = os.path.join("./model_api/","models/Neural Network")
    model_weights = os.listdir(path)
    model = MLPRegressor() # Create the Object
    model.load_state_dict(torch.load(os.path.join(path,model_weights[0]))) # Load the model weights
    model.eval() # Set the model to evaluation mode

    # Ensure x_input is a 2D tensor
    x_input = np.array(df[[input_column]], dtype=np.float32)
    if x_input.ndim == 0:
        x_input = x_input.reshape(1, 1)
    elif x_input.ndim == 1:
        x_input = x_input.reshape(-1, 1)

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