import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
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

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Adjust for class imbalance
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                          scale_pos_weight=scale_pos_weight, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
