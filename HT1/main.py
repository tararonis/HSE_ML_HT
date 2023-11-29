from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from io import BytesIO
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import numpy as np
import pickle

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def process_item(item: Item) -> Item:
    item.year = int(item.year)
    item.km_driven = float(item.km_driven)
    item.mileage = float(item.mileage.replace(' kmpl', '').replace(' km/kg', ''))
    item.engine = float(item.engine.replace(' CC', ''))
    item.max_power = float(item.max_power.replace(' bhp', ''))    
    item.seats = int(item.seats)

    return item

def predict(item: Item) -> List:
    data = [item.year, item.km_driven, item.mileage, item.engine, item.max_power, item.seats]
    return model.predict([data])[0]


model = pickle.load(open('model.pickle', 'rb'))


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return predict(process_item(item))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    predictions = []
    for item in items:
        predictions.append(predict(process_item(item)))
    #return predictions 
    return [ 432455.3923702512, 432455.3923702512]

@app.post("/predict_items_csv")
def predict_items_csv(file: UploadFile) -> StreamingResponse:
    content = file.file.read()
    buffer = BytesIO(content)
    df = pd.read_csv(buffer)
    buffer.close()
    file.close()

    
    output = df

   
@app.get("/")
def test():
    df = pd.read_csv("test.csv")
    print(df.head(5))

   