
import numpy as np
from sklearn.linear_model import SGDClassifier
from .metrics import compute_metrics

def run_federated_svm(X_train, y_train, X_test, y_test, num_clients, num_rounds):
    y_train = np.array(y_train)
    X_train = np.array(X_train.todense()) if hasattr(X_train, "todense") else np.array(X_train)

    global_coef = np.zeros(X_train.shape[1])
    global_intercept = 0

    for round_num in range(num_rounds):
        rng = np.random.default_rng(42 + round_num)
        class_0 = np.where(y_train == 0)[0]
        class_1 = np.where(y_train == 1)[0]
        rng.shuffle(class_0)
        rng.shuffle(class_1)

        split_0 = np.array_split(class_0, num_clients)
        split_1 = np.array_split(class_1, num_clients)
        client_indices = [np.concatenate((s0, s1)) for s0, s1 in zip(split_0, split_1)]

        local_coefs, local_intercepts = [], []
        for idx in client_indices:
            y_c = y_train[idx]
            if len(np.unique(y_c)) < 2:
                continue
            X_c = X_train[idx]
            clf_local = SGDClassifier(loss='hinge', max_iter=1, tol=None, random_state=42 + round_num)
            clf_local.fit(X_c, y_c)
            local_coefs.append(clf_local.coef_[0])
            local_intercepts.append(clf_local.intercept_[0])

        if not local_coefs:
            return {"precision": np.nan, "recall": np.nan, "f1_weighted": np.nan, "accuracy": np.nan, "roc_auc": np.nan}
        global_coef = np.mean(local_coefs, axis=0)
        global_intercept = np.mean(local_intercepts)

    clf_global = SGDClassifier(loss='hinge', random_state=42, class_weight='balanced')
    init_idx = [np.where(y_train == 0)[0][0], np.where(y_train == 1)[0][0]]
    clf_global.fit(X_train[init_idx], y_train[init_idx])
    clf_global.coef_ = global_coef.reshape(1, -1)
    clf_global.intercept_ = np.array([global_intercept])
    clf_global.classes_ = np.array([0, 1])

    y_pred = clf_global.predict(X_test)
    y_scores = clf_global.decision_function(X_test)
    return compute_metrics(y_test, y_pred, y_scores)

def run_federated_svm_noniid(X_train, y_train, X_test, y_test, num_clients, num_rounds):
    y_train = np.array(y_train)
    X_train = np.array(X_train.todense()) if hasattr(X_train, "todense") else np.array(X_train)

    global_coef = np.zeros(X_train.shape[1])
    global_intercept = 0

    for round_num in range(num_rounds):
        rng = np.random.default_rng(42 + round_num)
        class_0_idx = np.where(y_train == 0)[0]
        class_1_idx = np.where(y_train == 1)[0]
        rng.shuffle(class_0_idx)
        rng.shuffle(class_1_idx)

        client_indices = []
        for i in range(num_clients):
            if i % 2 == 0:
                size_0 = len(class_0_idx) // num_clients
                size_1 = int(size_0 * 0.2)
            else:
                size_1 = len(class_1_idx) // num_clients
                size_0 = int(size_1 * 0.2)

            client_0 = class_0_idx[:size_0]
            client_1 = class_1_idx[:size_1]
            client_indices.append(np.concatenate([client_0, client_1]))
            class_0_idx = class_0_idx[size_0:]
            class_1_idx = class_1_idx[size_1:]

        local_coefs, local_intercepts = [], []
        for idx in client_indices:
            y_c = y_train[idx]
            if len(np.unique(y_c)) < 2:
                continue
            X_c = X_train[idx]
            clf_local = SGDClassifier(loss='hinge', max_iter=1, tol=None, random_state=42 + round_num)
            clf_local.fit(X_c, y_c)
            local_coefs.append(clf_local.coef_[0])
            local_intercepts.append(clf_local.intercept_[0])

        if not local_coefs:
            return {"precision": np.nan, "recall": np.nan, "f1_weighted": np.nan, "accuracy": np.nan, "roc_auc": np.nan}
        global_coef = np.mean(local_coefs, axis=0)
        global_intercept = np.mean(local_intercepts)

    clf_global = SGDClassifier(loss='hinge', random_state=42, class_weight='balanced')
    init_idx = [np.where(y_train == 0)[0][0], np.where(y_train == 1)[0][0]]
    clf_global.fit(X_train[init_idx], y_train[init_idx])
    clf_global.coef_ = global_coef.reshape(1, -1)
    clf_global.intercept_ = np.array([global_intercept])
    clf_global.classes_ = np.array([0, 1])

    y_pred = clf_global.predict(X_test)
    y_scores = clf_global.decision_function(X_test)
    return compute_metrics(y_test, y_pred, y_scores)
