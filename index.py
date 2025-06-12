from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.exponential_smoothing.ets import ETSResults
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import pandas as pd
from typing import *


app = FastAPI()

class Sale(BaseModel):
    date: str
    quantity: int

class PredictRequest(BaseModel):
    sales: List[Sale]
    period: int

additive_configs = [
    {'error': 'add', 'trend': None, 'seasonal': None, 'damped_trend': False},
    {'error': 'add', 'trend': None, 'seasonal': 'add', 'damped_trend': False},
    {'error': 'add', 'trend': 'add', 'seasonal': None, 'damped_trend': False},
    {'error': 'add', 'trend': 'add', 'seasonal': None, 'damped_trend': True},
    {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': False},
    {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
]

@app.post("/predict")
def predict(request: PredictRequest):

    # Load and preprocess
    df = pd.DataFrame(sale.model_dump() for sale in request.sales)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')

    df_train = df[:-3]
    df_test = df[-3:]

    # Use the Series, not flattened NumPy array
    df_train_one_d = df_train["quantity"]
    df_test_one_d = df_test["quantity"]
  
    # start find best model configurations
    best_mape = float("inf")
    best_model = None
    y_pred = None

    for config in additive_configs:  # or valid_configs if you want to include multiplicative
        try:
            model = ETSModel(
                endog=df_train_one_d, 
                error=config["error"], 
                trend=config["trend"], 
                seasonal=config["seasonal"],
                damped_trend=config["damped_trend"], 
                seasonal_periods=12,
                initialization_method="estimated",
            )


            fit = model.fit(maxiter=100000)
            y_pred = fit.predict(start=df_test_one_d.index[0], end=df_test_one_d.index[-1])
            print(f"Config: {config}, MAPE: {mean_absolute_percentage_error(df_test_one_d, y_pred)}")

            mape = mean_absolute_percentage_error(df_test_one_d, y_pred)

            if mape < best_mape:
                best_mape = mape
                best_model = fit
            
        except Exception as e:
            print(f"Error with config {config}: {e}")
    

    if best_model is None:
        return {"error": "No suitable model found for the provided sales data."}

    latest_date = df.index[-1]
    future_dates = pd.date_range(start=latest_date + pd.DateOffset(months=1), 
                                  periods=request.period, freq='MS')
    
    future_predictions = best_model.predict(start=future_dates[0], end=future_dates[-1])
    predictions_df = pd.DataFrame({
        'date': future_dates,
        'total': future_predictions
    })
    predictions_df.set_index('date', inplace=True)
    predictions_list = predictions_df.reset_index().to_dict(orient='records')

    return {
        "predictions": predictions_list
    }
