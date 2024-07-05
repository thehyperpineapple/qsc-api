import torch
import json
import os
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file

try:
    tokenizer = DistilBertTokenizer.from_pretrained('./model_api/models/Text Classifier/', local_files_only=True)

    config_path = "./model_api/models/Text Classifier/config.json"
    # Load the config.json file
    with open(config_path, 'r') as f:
        config = json.load(f)

except Exception as error:
    print(error)

# Load the model
model_dir = './model_api/models/Text Classifier/'
#  Verify the model directory exists
# if not os.path.exists(model_dir):
#     raise OSError(f"The directory {model_dir} does not exist.")

try:
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir, local_files_only=True)
    
    # Load the config.json file
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The config file was not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

except Exception as error:
    print(f"Error loading tokenizer or config: {error}")

try:
    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    # Load the weights from the safetensors file
    weights_path = os.path.join(model_dir, 'model.safetensors')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"The weights file was not found at {weights_path}")
    
    weights = load_file(weights_path)
    model.load_state_dict(weights, strict=False)
    model.eval()

except Exception as error:
    print(f"Error loading model or weights: {error}")


# Mapping
threat_mapping = {
    0: 'Authentication and Access Anomalies', 
    1: 'File and Search Activity', 
    2:'Impersonation and Phishing', 
    3:'Threats and Malicious Activity'
    }

# Put the model in evaluation mode


def classify_text_batch(inputs):
    tokenized_inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    predicted_classes = torch.argmax(outputs.logits, dim=1).tolist()
    return [threat_mapping[pred] for pred in predicted_classes]

def bulk_classify_text(df, batch_size=32, max_workers=4):
    descriptions = df['Description'].tolist()
    classified_descriptions = []

    def process_batch(batch):
        return classify_text_batch(batch)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch))
        
        for future in futures:
            classified_descriptions.extend(future.result())
    
    df['Classified Description'] = classified_descriptions
    return df

# df = pd.read_csv("C:/Users/Adit Prasad/Downloads/Sample Final_Input.csv", encoding = 'cp1252')
# df = bulk_classify_text(df)
# print(df)