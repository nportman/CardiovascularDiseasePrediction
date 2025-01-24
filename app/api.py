from fastapi import FastAPI, UploadFile, File
import pandas as pd
from models.simple_model import predict_rule_based
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.post("/predict")
def predict(slope: int):
    if slope not in [0, 1, 2, 3]:
        return {"error": "Invalid slope value. Please enter a value between 0 and 3."}
    
    feature_vector = pd.Series({'slope': slope})
    prediction = predict_rule_based(feature_vector)
    
    if prediction is not None:
        if prediction == 0.0:
            return {"prediction": "Patient has no cardiovascular disease."}
        else:
            return {"prediction": "Patient has cardiovascular disease."}
    else:
        return {"error": "Unable to make a prediction."}

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file)
        
        if 'slope' not in df.columns:
            return {"error": "The uploaded CSV must contain a 'slope' column."}
        
        predictions = df['slope'].apply(lambda x: predict_rule_based(pd.Series({'slope': x})))
        df['prediction'] = predictions
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False).encode('utf-8')
        return StreamingResponse(io.BytesIO(csv), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    else:
        return {"error": "File type not supported. Please upload a CSV file."}
