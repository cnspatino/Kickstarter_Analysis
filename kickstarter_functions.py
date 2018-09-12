import pandas as pd
import numpy as np

def get_duration(df, launched, deadline):
    """
    Function that gets duration (in number of days) of a project based on launched and deadline dates.
    
    INPUTS: dataframe, launched column (string), deadline column (string)
    OUTPUT: modified dataframe now with duration column
    """
    
    # convert deadline and launched columns from strings to datetime objects
    df[deadline] = pd.to_datetime(df[deadline])
    df[launched] = pd.to_datetime(df[launched]).dt.date #drops time part
    df[launched] = pd.to_datetime(df[launched]) #convert back to datetime object
    
    # create duration column
    df['duration'] = df[deadline] - df[launched] #creates timedelta object
    # convert from timedelta to number of days
    df['duration'] = df['duration'].apply(lambda x: x.days)
    
    return df

def fill_missing_countries(df):
    # fill missing country values using currency (except for when currency is euro)
    df.country = np.where((df.country == 'N,0"') & (df.currency != 'EUR'), 
                             df.currency.str[:-1], df.country)
    # convert remaining values of N,0" to null
    df.country = np.where(df.country == 'N,0"', None, df.country)

    return df



