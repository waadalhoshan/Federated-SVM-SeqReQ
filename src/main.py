
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_loader import load_data
from src.centralized_svm import train_centralized_svm
from src.federated_svm import run_federated_svm, run_federated_svm_noniid

def main():
    data_path = os.path.join("dataset", "seqreq.csv")
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["RequirementText"], df["label_enc"],
        test_size=0.3, random_state=42, stratify=df["label_enc"]
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("📊 Running Centralized SVM...")
    central_metrics = train_centralized_svm(X_train_tfidf, y_train, X_test_tfidf, y_test)
    print("Centralized Metrics:", central_metrics)

    client_list = [2, 3, 4, 5]
    round_list = [1, 3, 5, 10]
    results_iid = []
    results_noniid = []

    print("🤖 Running Federated SVM experiments...")
    for clients in client_list:
        for rounds in round_list:
            res_iid = run_federated_svm(X_train_tfidf, y_train.values, X_test_tfidf, y_test.values, clients, rounds)
            res_iid.update({"Setting": "IID", "Clients": clients, "Rounds": rounds})
            results_iid.append(res_iid)

            res_noniid = run_federated_svm_noniid(X_train_tfidf, y_train.values, X_test_tfidf, y_test.values, clients, rounds)
            res_noniid.update({"Setting": "non-IID", "Clients": clients, "Rounds": rounds})
            results_noniid.append(res_noniid)

    results_df = pd.DataFrame(results_iid + results_noniid)
    results_df.to_csv(os.path.join("results", "federated_svm_results.csv"), index=False)
    print("✅ Results saved to results/federated_svm_results.csv")

if __name__ == "__main__":
    main()
