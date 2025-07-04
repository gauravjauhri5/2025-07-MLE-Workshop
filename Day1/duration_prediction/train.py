#!/usr/bin/env python
# coding: utf-8

from datetime import date
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import argparse
from loguru import logger

def read_dataframe(filename: str):
    """
    Reads a Parquet file into a pandas DataFrame, calculates trip duration in minutes, 
    filters out trips outside the 1-60 minute range, and converts specified columns to string.

    Parameters:
    ----------
    filename : str
        Path to the Parquet file containing trip data.

    Returns:
    -------
    pd.DataFrame
        A cleaned DataFrame with an added 'duration' column (in minutes), 
        filtered for valid trip durations, and with categorical location IDs 
        converted to string type.
    """
    logger.info(f"loading file: {filename}")
    try:
        df = pd.read_parquet(filename)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        logger.info(f"{filename} had {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"error reading: {filename}")
        logger.error(e)
        raise


def train(train_date: date, val_date: date, out_path: str):
    """
    Trains a linear regression model to predict trip duration based on NYC green taxi trip data.

    Downloads training and validation datasets based on input dates, processes them,
    fits a model using DictVectorizer and LinearRegression, evaluates it using RMSE, 
    and saves the trained pipeline to a file.

    Parameters
    ----------
    train_date : date
        The date representing the training dataset's month and year.
    val_date : date
        The date representing the validation dataset's month and year.
    out_path : str
        Path where the trained model pipeline will be saved (as a pickle file).

    Returns
    -------
    None
        The function saves the trained model pipeline to the specified path and prints RMSE.
    """
    base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = base_url.format(year=train_date.year, month=train_date.month)
    val_url = base_url.format(year=val_date.year, month=val_date.month)
    
    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    target = 'duration'
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv = DictVectorizer()
    lr = LinearRegression()
    pipeline = make_pipeline(dv, lr)
    
    pipeline.fit(train_dicts, y_train)
    y_pred = pipeline.predict(val_dicts)

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"{mse=}")

    logger.info(f"writing model into {out_path}")
    with open(out_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    
    return mse