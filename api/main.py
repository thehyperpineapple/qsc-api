from typing import Union
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from model_api import model_predictions, utility_functions
import asyncio
from io import BytesIO
import pandas as pd
import openpyxl 
import csv 



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'], #Allow all origins
    allow_credentials=True,
    allow_methods = ['*'],
    allow_headers=['*']
)

@app.get("/")
async def read_root():
    return {"Hello":"World"}

@app.get("/single_response/{total_count}")
async def return_single_response(total_count):
    results = model_predictions.single_predictions(total_count)
    return results



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Debugging: Print the file name and first few bytes
    print(f"Received file: {file.filename}")
    print(f"File size: {len(contents)} bytes")

   # Read the file into a DataFrame
    try:
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        return {"error": str(e)}
    
    # Debugging: Print the DataFrame shape and first few rows
    print(f"DataFrame shape: {df.shape}")
    print(df.head())

    new_df = utility_functions.analyze_descriptions(df)
    new_df = model_predictions.bulk_prediction(new_df)
    new_df = model_predictions.bulk_deep_learning(new_df)

    print(f"DataFrame shape: {new_df.shape}")
    print(new_df.head())

    # Save the DataFrame back to a file
    output = BytesIO()
    if file.filename.endswith('.xlsx'):
        new_df.to_excel(output, index=False)
    else:
        new_df.to_csv(output, index=False)
    
    output.seek(0)
    return StreamingResponse(output, media_type='application/octet-stream', headers={"Content-Disposition": f"attachment; filename={file.filename}"})

# Post the file to the FastAPI server
# files = {"file": uploaded_file.getvalue()}
# response = requests.post("http://127.0.0.1:8000/uploadfile/", files=files)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)