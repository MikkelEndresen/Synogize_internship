from snowflake.snowpark import Session
import numpy as np
import pandas as pd

def create_session():
    session = Session.builder.configs({
        'user': 'mikkel',
        'password': 'qPz4F6b,Bdkx@v&',
        'account': 'DFBRTSW-VI83989',
        'database': 'ML',
        'schema': 'retail_store'
    }).create()
    return session

def mape(actual, pred): 
    """
    Parameters:
        - Actual
        - Predicted
    Returns:
        - MAPE
    """

    actual, pred = np.array(actual), np.array(pred)
    return round(np.mean(np.abs((actual - pred) / actual)) * 100, 2)


def get_time_series_1():

    session = create_session()

    df = session.table("company_revenue_time_series_1") #with cutoff
    df = df.to_pandas()

    df = df.rename(columns={'DATE': 'ds', 'REVENUE': 'y'})
    df = df[['ds', 'y']] # drop everything but these two if there are any

    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    df = df.reset_index(drop=True)

    print(df.head())
    print(df.info())

    return df

def get_time_series_2():

    session = create_session()

    df = session.table("company_revenue_time_series_2") #with cutoff
    df = df.to_pandas()

    df = df.rename(columns={'DATE': 'ds', 'REVENUE': 'y'})
    df = df[['ds', 'y']] # drop everything but these two if there are any

    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    df = df.reset_index(drop=True)

    print(df.head())
    print(df.info())

    return df

def z_score_outlier(df, z_score):
    mean = df['y'].mean()
    std = df['y'].std()

    z_scores = (df['y'] - mean) / std

    threshold = z_score
    outlier_index = np.where(np.abs(z_scores) > threshold)[0]
    
    print("Dropping "+str(len(outlier_index)) + " rows.")
    
    df.drop(index=outlier_index,inplace=True)

    return df

def download_csv():
    session = create_session()

    df = session.table("company_revenue_time_series_1") #with cutoff
    df = df.to_pandas()

    file_path = "company_revenue_w_cutoff.csv"

    df.to_csv(file_path, mode='a', header=False, index=False)
    
    return "success"


def csv_to_df_1():

    df = pd.read_csv("data/company_revenue_w_cutoff.csv")

    return df

def csv_to_df_2():

    df = pd.read_csv("data/company_revenue.csv")

    return df

