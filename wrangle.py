import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def clean_df():
    # acquire df
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    
    # lowercase column and row names
    df.columns = df.columns.str.lower()
    
    # lowercase rows
    df['residence_type'] = df['residence_type'].apply(str.lower)
    df['work_type'] = df['work_type'].apply(str.lower)
    df['gender'] = df['gender'].apply(str.lower)
    df['smoking_status'] = df['smoking_status'].apply(str.lower)
    
    # replace nulls with average bmi 
    df.bmi.fillna(28.9, inplace = True)
    
    # replace Yes: 1 and No: 0
    df.ever_married = df.ever_married.replace("Yes", 1)
    df.ever_married = df.ever_married.replace("No", 0)
    
    # drop "other" row
    df.drop(df[df.gender == "other"].index, inplace=True)

    # drop "id"
    df = df.drop(columns=["id"])

    # create column if patient is a child from work type
    # df['is_child'] = (df['work_type'] == "children").astype(int)
    df['is_child'] = df['age'].apply(lambda age: 1 if age < 18 else 0)
    
    # get dummies for columns
    dummy_df = pd.get_dummies(df[["gender",
                       "work_type",
                       "residence_type",
                       "smoking_status"]],
                       drop_first=True)
    # columns to keep from original df
    num_df = df[["age", "is_child", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi", "stroke"]]

    # join dummy df and df
    model_df = pd.concat([dummy_df, num_df], axis=1)
    df= pd.concat([df, dummy_df], axis=1)
    
    return df, model_df

def split_data(df, target_variable):
    '''
    Takes in two arguments the dataframe name and the ("target_variable" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order.
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify= df[target_variable])
    train, validate = train_test_split(train, #second split
                                    test_size=.25, 
                                    random_state=123, 
                                    stratify=train[target_variable])
    return train, validate, test