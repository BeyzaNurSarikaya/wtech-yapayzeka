from fastapi import FastAPI
from pydantic import BaseModel # Fonksiyonlara gelen parametreleri kontrol etmek için
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model # Modeli yükleme

# Define the FastAPI app
app = FastAPI()


class ObesityModel(BaseModel):
    Age: float
    Gender: object
    Height: float
    Weight: float
    CALC: object
    FAVC: object
    FCVC: float
    NCP: float
    SCC: object
    SMOKE: object
    CH20: float
    family_history_with_overweight: object
    FAF: float
    TUE: float
    CAEC: object
    MTRANS: object


# obesity model deployment
@app.post("/predict_obesity/")
async def predict_obesity(item: ObesityModel):
    load_ann = load_model('C:/Users/HP/Documents/sites/project1/Wtech/models/obesity.h5')
    data = pd.DataFrame([item.dict()])

    # LabelEncoder'ı oluştur ve uygun sütunlara uygula
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])

    prediction = load_ann.predict(data)
    print(prediction)

    # [[0.5200778  0.43981093 0.0401112 ]] max değeri al
    prediction = prediction[0]
    print(prediction)

    Nobeyesdad = prediction.argmax()
    print(Nobeyesdad)

    if Nobeyesdad == 0:
        Nobeyesdad = "Insufficient_Weight"
    elif Nobeyesdad == 1:
        Nobeyesdad = "Normal_Weight"
    elif Nobeyesdad == 2:
        Nobeyesdad = "Obesity_Type_I"
    elif Nobeyesdad == 3:
        Nobeyesdad = "Obesity_Type_II"
    elif Nobeyesdad == 4:
        Nobeyesdad = "Obesity_Type_III"
    elif Nobeyesdad == 5:
        Nobeyesdad = "Overweight_Level_I"
    else:
        Nobeyesdad = "Overweight_Level_II"
    return {"Nobeyesdad": Nobeyesdad}


# Hello world endpoint
@app.get("/")
async def root():
    return {"message": "Hello, World!"}
# python -m uvicorn obesity:app --reload

# http://127.0.0.1:8000/docs