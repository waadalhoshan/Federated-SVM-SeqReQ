
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def train_centralized_svm(X_train, y_train, X_test, y_test):
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.decision_function(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_scores)
    }
    return metrics
