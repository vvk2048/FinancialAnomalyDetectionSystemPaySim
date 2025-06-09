from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_model(y_true, y_pred, scores):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("ROC AUC Score:", roc_auc_score(y_true, scores))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
