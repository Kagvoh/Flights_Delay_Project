from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

app = FastAPI(
    title='Flight Delay Prediction API',
    description='Predict flight delay using ML models',
    version='1.0'
)


# LOAD MODELS

logistic_model = joblib.load('models/logistic.pkl')
dt_model = joblib.load('models/decision_tree.pkl')
xgb_model = joblib.load('models/xgboost.pkl')

scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

route_distance = joblib.load('models/route_distance.pkl')


# INPUT SCHEMA
class FlightInput(BaseModel):
    Month: int
    DayofMonth: int
    DayOfWeek: int

    CRSDepTime: int
    CRSArrTime: int
    
    UniqueCarrier: str
    Origin: str
    Dest: str

    model_name: str


# FEATURE ENGINEERING
def convert_time_to_float(time_str):
    return time_str // 100 + (time_str % 100 )/ 60

def calculate_elapsed_time(dep_time, arr_time):

    dep_hour = dep_time // 100
    dep_min = dep_time % 100
    
    arr_hour = arr_time // 100
    arr_min = arr_time % 100

    dep_total = dep_hour*60 + dep_min
    arr_total = arr_hour*60 + arr_min

    elapsed_time = arr_total - dep_total

    if(elapsed_time < 0):
        elapsed_time += 1440
        
    return elapsed_time

def valid_time(t):

    hour = t // 100
    minute = t % 100

    return (0 <= hour <= 23 and 0 <= minute <= 59)

def get_period(time):
    if time < 6:
        return 'Night/Early Morning'
    elif time < 12:
        return 'Morning'
    elif time < 18:
        return 'Afternoon'
    else:
        return 'Evening'


# PREPROCESS DATA

def preprocess_input(data):
    
    elapsed_time = calculate_elapsed_time(
        data.CRSDepTime,
        data.CRSArrTime
    )

    dep_time = convert_time_to_float(data.CRSDepTime)

    arr_time = convert_time_to_float(data.CRSArrTime)
    
    dep_period = get_period(dep_time)
    arr_period = get_period(arr_time)

    # Distance auto lookup
    distance = route_distance.get(
        (data.Origin, data.Dest),
        500
    )
    
    df = pd.DataFrame([
        {
            'Month': data.Month,
            'DayofMonth': data.DayofMonth,
            'DayOfWeek': data.DayOfWeek,
            'CRSDepTime': dep_time,
            'CRSArrTime': arr_time,
            'CRSElapsedTime': elapsed_time,
            'Distance': distance,
            'UniqueCarrier': data.UniqueCarrier,
            'Origin': data.Origin,
            'Dest': data.Dest,
            'DepTimePeriod': dep_period,
            'ArrTimePeriod': arr_period
        }
    ])

    numeric_features = [
        'Month',
        'DayofMonth',
        'DayOfWeek',
        'CRSDepTime',
        'CRSArrTime',
        'CRSElapsedTime',
        'Distance'
    ]

    categorical_features = [
        'UniqueCarrier',
        'Origin',
        'Dest',
        'DepTimePeriod',
        'ArrTimePeriod'
    ]

    X_num = scaler.transform(df[numeric_features])
    X_cat = encoder.transform(df[categorical_features])

    X = hstack([X_num, X_cat])

    return X, distance, dep_period, arr_period, elapsed_time


# SELECT MODEL FOR PREDICTING

def get_model(model_name):

    if model_name == 'Logistic Regression':
        return logistic_model

    elif model_name == 'Decision Tree':
        return dt_model

    elif model_name == 'XGBoost':
        return xgb_model

    return xgb_model


# API ROUTE

@app.get('/')
def home():
    return {
        'message': 'Flights Delay Prediction API Running'
    }

@app.post('/predict')
def predict(data: FlightInput):
    
    if not valid_time(data.CRSDepTime):
        return {'error': 'Invalid Departure Time'}
    
    if not valid_time(data.CRSArrTime):
        return {'error': 'Invalid Arrival Time'}
        
    X, distance, dep_period, arr_period, elapsed_time = preprocess_input(data)

    model = get_model(data.model_name)

    probability = model.predict_proba(X)[0][1]
    
    if data.model_name == 'XGBoost':
        threshold = 0.7

    elif data.model_name == 'Decision Tree':
        threshold = 0.5

    else:
        threshold = 0.5
     
    prediction = int(probability >= threshold)
    
    return {
        'prediction': int(prediction),
        'probability_delay': round(float(probability), 4),
        'distance': round(float(distance), 2),
        'CRS_departure_period': dep_period,
        'CRS_arrival_period': arr_period,
        'CRS_elapsed_time': elapsed_time,
        'message': 'Delayed' if prediction == 1 else 'On Time'
    }
