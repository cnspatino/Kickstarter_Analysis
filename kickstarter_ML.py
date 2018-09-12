import pandas as pd
import numpy as np
import kickstarter_functions as kf
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def clean_data(df):
    """
    Takes in dataframe and performs the necessary pre-processing steps before using it for a
    logistic regression model. Performs all steps but scaling step.
    
    INPUT: dataframe after unfinished/undefined entries are dropped
    OUTPUT: modified dataframe after pre-processing/cleaning. 
    """
    
    # fill missing country values using currency (except for when currency is euro)
    df = kf.fill_missing_countries(df)
    
    # create duration column
    df = kf.get_duration(df, 'launched', 'deadline')

    # drop unwanted columns
    drop_cols = ['ID','name','category','currency','goal','pledged','usd pledged', 
                 'launched','deadline', 'usd_pledged_real']
    df.drop(drop_cols, axis=1, inplace=True)
    
    # map successful state to 1 and failed to 0
    df.state = df.state.map({'successful':1, 'failed':0})
    df.rename({'state':'succeeded'}, axis=1, inplace=True) #rename column
    
    # create dummy variables
    df = pd.get_dummies(df)
    
    # transform skewed numerical values
    skewed = ['backers','usd_goal_real']
    df[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
    
    return df

def scale_data(df, columns):
    """
    Applies a RobustScaler to the specified columns of a dataframe.

    INPUTS: dataframe, numerical columns to apply the scaler to (list of strings)
    OUTPUT: scaled dataframe

    """

    # Initialize a scaler, then apply it to the numerical features
    scaler = RobustScaler() # most values will fall in [-2,3]
    df_scaled = pd.DataFrame(data=df)
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    return df_scaled

def build_lr_model(df, test_size=0.3, random_state=42):
    """
    Builds a logistic regression model for kickstarter data to predict
    project outcomes.

    INPUTS: dataframe, test size (optional), random_state (optional)
    OUTPUTS: logistic regression model, predicted y values for test set,
             predicted y values for training set

    """

    y = df_scaled['succeeded'] # set target variable
    X = df_scaled.drop(['succeeded'], axis=1) # set features

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

    # Instantiate model
    lr_mod = LogisticRegression()

    # fit model to training data
    lr_mod.fit(X_train, y_train)

    # get predictions for test set and training set
    ytest_preds = lr_mod.predict(X_test)
    ytrain_preds = lr_mod.predict(X_train)

    return lr_mod, ytest_preds, ytrain_preds


