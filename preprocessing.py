import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_training(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[:,'TotalCharges'] = df.loc[:,'TotalCharges'].replace(np.nan,0)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(str)
    senior_map = {'0': 'No', '1': 'Yes'}
    df['SeniorCitizen'] = df['SeniorCitizen'].map(senior_map)
    df = df.dropna()
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])
    X = df.iloc[:,:-1]
    y = df.iloc[ :, -1:]
    X = X.drop(['customerID'], axis = 1)

    gender_map = {'Female': 0, 'Male': 1}
    yes_or_no_map = {'No': 0, 'Yes': 1} #seniorcitizen, partner, dependents, phoneservice, paperlessbilling
    multiplelines_map = {'No phone service': -1, 'No': 0, 'Yes': 1}
    internetservice_map = {'No': -1, 'DSL': 0, 'Fiber optic': 1}
    add_netservices_map = {'No internet service': -1, 'No': 0, 'Yes': 1} #onlinesecurity, onlinebackup, deviceprotection,techsupport,streaming services
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paymentmethod_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
    
    X['gender'] = X['gender'].map(gender_map).astype('int')
    X['Partner'] = X['Partner'].map(yes_or_no_map).astype('int')
    X['SeniorCitizen'] = X['SeniorCitizen'].map(yes_or_no_map).astype('int')
    X['Dependents'] = X['Dependents'].map(yes_or_no_map).astype('int')
    X['PhoneService'] = X['PhoneService'].map(yes_or_no_map).astype('int')
    X['MultipleLines'] = X['MultipleLines'].map(multiplelines_map).astype('int')
    X['InternetService'] = X['InternetService'].map(internetservice_map).astype('int')
    X['OnlineSecurity'] = X['OnlineSecurity'].map(add_netservices_map).astype('int')
    X['OnlineBackup'] = X['OnlineBackup'].map(add_netservices_map).astype('int')
    X['DeviceProtection'] = X['DeviceProtection'].map(add_netservices_map).astype('int')
    X['TechSupport'] = X['TechSupport'].map(add_netservices_map).astype('int')
    X['StreamingTV'] = X['StreamingTV'].map(add_netservices_map).astype('int')
    X['StreamingMovies'] = X['StreamingMovies'].map(add_netservices_map).astype('int')
    X['Contract'] = X['Contract'].map(contract_map).astype('int')
    X['PaperlessBilling'] = X['PaperlessBilling'].map(yes_or_no_map).astype('int')
    X['PaymentMethod'] = X['PaymentMethod'].map(paymentmethod_map).astype('int')
    return X,y

def preprocess_test(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[:,'TotalCharges'] = df.loc[:,'TotalCharges'].replace(np.nan,0)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(str)
    senior_map = {'0': 'No', '1': 'Yes'}
    df['SeniorCitizen'] = df['SeniorCitizen'].map(senior_map)
    df = df.dropna()
    X = df
    X = X.drop(['customerID'], axis = 1)
    gender_map = {'Female': 0, 'Male': 1}
    yes_or_no_map = {'No': 0, 'Yes': 1} #seniorcitizen, partner, dependents, phoneservice, paperlessbilling
    multiplelines_map = {'No phone service': -1, 'No': 0, 'Yes': 1}
    internetservice_map = {'No': -1, 'DSL': 0, 'Fiber optic': 1}
    add_netservices_map = {'No internet service': -1, 'No': 0, 'Yes': 1} #onlinesecurity, onlinebackup, deviceprotection,techsupport,streaming services
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paymentmethod_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
    
    X['gender'] = X['gender'].map(gender_map).astype('int')
    X['Partner'] = X['Partner'].map(yes_or_no_map).astype('int')
    X['SeniorCitizen'] = X['SeniorCitizen'].map(yes_or_no_map).astype('int')
    X['Dependents'] = X['Dependents'].map(yes_or_no_map).astype('int')
    X['PhoneService'] = X['PhoneService'].map(yes_or_no_map).astype('int')
    X['MultipleLines'] = X['MultipleLines'].map(multiplelines_map).astype('int')
    X['InternetService'] = X['InternetService'].map(internetservice_map).astype('int')
    X['OnlineSecurity'] = X['OnlineSecurity'].map(add_netservices_map).astype('int')
    X['OnlineBackup'] = X['OnlineBackup'].map(add_netservices_map).astype('int')
    X['DeviceProtection'] = X['DeviceProtection'].map(add_netservices_map).astype('int')
    X['TechSupport'] = X['TechSupport'].map(add_netservices_map).astype('int')
    X['StreamingTV'] = X['StreamingTV'].map(add_netservices_map).astype('int')
    X['StreamingMovies'] = X['StreamingMovies'].map(add_netservices_map).astype('int')
    X['Contract'] = X['Contract'].map(contract_map).astype('int')
    X['PaperlessBilling'] = X['PaperlessBilling'].map(yes_or_no_map).astype('int')
    X['PaymentMethod'] = X['PaymentMethod'].map(paymentmethod_map).astype('int')
    return X

