from fastapi import FastAPI, Depends
import pandas as pd
import uvicorn
from pydantic import BaseModel
import joblib
from preprocessing import preprocess_test


classes = {
    0: 'No',
    1: 'Yes',
}

class Feature_type(BaseModel):
        customerID : str = 'id'
        gender : str =	'Male'
        SeniorCitizen : int = 0
        Partner	: str = 'Yes'
        Dependents : str ='Yes'
        tenure : int = 69
        PhoneService : str ='Yes'
        MultipleLines: str = 'No'
        InternetService: str = 'Fiber optic'
        OnlineSecurity: str = 'No'
        OnlineBackup: str = 'No'
        DeviceProtection: str = 'No'
        TechSupport: str ='Yes'
        StreamingTV: str ='Yes'
        StreamingMovies	: str ='Yes'
        Contract: str =	'Two year'
        PaperlessBilling: str =	'Yes'
        PaymentMethod	: str = 'Credit card (automatic)'
        MonthlyCharges: float =	95.2
        TotalCharges: float =	6671.7

model = joblib.load("cat_model.pkl")

app = FastAPI(title="API1", description="with FastAPI by Daniele Grotti", version="1.0")


@app.get("/", status_code=200)
def home():
    return {" ---->          http://localhost:8000/docs     <----------"}


@app.get("/predict", status_code=200)
async def predict_get(data: Feature_type= Depends()):              # depends() input nelle celle
    try:
        df = pd.DataFrame(data)
        df = df.T
        df.columns = df.iloc[0]
        df = df[1:]
        print(df)
        data =  preprocess_test(df)
        print(data)
        y_pred = list(map(lambda x: classes[x], model.predict(data).tolist()))[0]
        return {"prediction":y_pred}
    except:
        return {"prediction": "error"}

@app.post("/predict", status_code=200)
async def predict_post(data: Feature_type= Depends()):              # depends() input nelle celle
    try:
        df = pd.DataFrame(data)
        df = df.T
        df.columns = df.iloc[0]
        df = df[1:]
        print(df)
        data =  preprocess_test(df)
        print(data)
        y_pred = list(map(lambda x: classes[x], model.predict(data).tolist()))[0]
        return {"prediction":y_pred}
    except:
        return {"prediction": "error"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
