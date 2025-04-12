from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

dummy_data = [
    {"date": "2023-01", "quantity": 1},
    {"date": "2023-02", "quantity": 20},
    {"date": "2023-03", "quantity": 8},
    {"date": "2023-04", "quantity": 4},
    {"date": "2023-05", "quantity": 5},
    {"date": "2023-06", "quantity": 60},
    {"date": "2023-07", "quantity": 7},
    {"date": "2023-08", "quantity": 2},
    {"date": "2023-09", "quantity": 30},
    {"date": "2023-10", "quantity": 12},
    {"date": "2023-11", "quantity": 11},
    {"date": "2023-12", "quantity": 12},
    {"date": "2024-01", "quantity": 13},
    {"date": "2024-02", "quantity": 14},
    {"date": "2024-03", "quantity": 21},
    {"date": "2024-04", "quantity": 15},
    {"date": "2024-05", "quantity": 17},
    {"date": "2024-06", "quantity": 10},
    {"date": "2024-07", "quantity": 1},
    {"date": "2024-08", "quantity": 2},
    {"date": "2024-09", "quantity": 21},
    {"date": "2024-10", "quantity": 22},
    {"date": "2024-11", "quantity": 23},
    {"date": "2024-12", "quantity": 23},
    {"date": "2025-01", "quantity": 22},
    {"date": "2025-02", "quantity": 26},
    {"date": "2025-03", "quantity": 11},
    {"date": "2025-04", "quantity": 8},
    {"date": "2025-05", "quantity": 9},
    {"date": "2025-06", "quantity": 3},
]

alpha: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta: list[float | None] = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gamma: list[float | None] = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

seasonal_periods = 4

class Item(BaseModel):
    date: str
    quantity: int

class Sales(BaseModel):
    sales: list[Item] = dummy_data

@app.get("/predict")
def predict(sales: Sales):
    series = pd.Series([item['quantity'] for item in sales.sales], index=pd.to_datetime([item['date'] for item in sales.sales]))
    
    


    return {"predicted_sales": [100, 200, 300]}

