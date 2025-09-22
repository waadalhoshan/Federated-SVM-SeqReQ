
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def compute_metrics(y_true, y_pred, y_scores):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores)
    }
