import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['amount_log'] = np.log1p(df['amount'])
    df = df.drop(columns=['nameOrig', 'nameDest'])

    features = ['amount_log', 'step', 'isFlaggedFraud',
                'oldbalanceOrg','newbalanceOrig',
                'oldbalanceDest','newbalanceDest']

    X = df[features].values
    y = df['isFraud'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
