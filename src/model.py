from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_scaled, contamination):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X_scaled)
    return model

def predict_anomalies(model, X_scaled):
    scores = model.decision_function(X_scaled)
    preds = (model.predict(X_scaled) == -1).astype(int)
    return scores, preds
