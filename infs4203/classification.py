from dataset import INFS4203Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from utils import get_data_dir, standardize_test_data
import pandas as pd

class KNNClassifier:
    def __init__(self, hyperparams: Dict[str, List[Any]]):
        self.knn = KNeighborsClassifier()
        self.hyperparams = hyperparams

    def fit(self, X, y):
        grid_search = GridSearchCV(
            self.knn,
            self.hyperparams,
            cv=10,
            verbose=1,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid_search.fit(X, y)
        self.knn = grid_search.best_estimator_

        print("KNN best params: ", grid_search.best_params_)
        print("KNN best score: ", grid_search.best_score_)
    
    def score(self, X, y):
        predicted_labels = self.knn.predict(X)
        return f1_score(y, predicted_labels, average="macro")

class KMeansClassifier:
    def __init__(self, hyperparams: Dict[str, List[Any]]):
        self.kmeans = KMeans(n_init=1)
        self.hyperparams = hyperparams
        self.cluster_to_label_mapping = None

    def fit(self, X, y):
        # Perform the grid search
        grid_search = GridSearchCV(
            self.kmeans,
            self.hyperparams,
            cv=10,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X)
        print("KMeans best params: ", grid_search.best_params_)
        self.kmeans = grid_search.best_estimator_
        cluster_assignments = self.kmeans.labels_
        
        # Create a mapping from cluster ID to most frequent y value
        self.cluster_to_label_mapping = {}
        for cluster_id in set(cluster_assignments):
            true_labels_in_cluster = [y[i] for i, label in enumerate(cluster_assignments) if label == cluster_id]
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            self.cluster_to_label_mapping[cluster_id] = most_common_label

    def predict(self, X):
        clusters = self.kmeans.predict(X)
        return [self.cluster_to_label_mapping[cluster] for cluster in clusters]

    def score(self, X, y):
        predicted_labels = self.predict(X)
        return f1_score(y, predicted_labels, average="macro")


if __name__ == '__main__':
    # ---------- Load the dataset --------------------
    dataset = INFS4203Dataset("train.csv", preprocessing=True)
    data = dataset.std_normalized_df

    X = data[dataset.feature_columns].values
    y = data.Label.values

    hyperparams_knn = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    hyperparams_kmeans = {
        "n_clusters": [3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18],
    }

    # ---------- KNN --------------------
    knn = KNNClassifier(hyperparams_knn)
    knn.fit(X, y)

    # ---------- KMeans --------------------
    kmeans = KMeansClassifier(hyperparams_kmeans)
    kmeans.fit(X, y)
    print("KMeans score: ", kmeans.score(X, y))




    # ----------------- Testing the algorithms on completely unseen data --------------
    testset = INFS4203Dataset("train2.csv", preprocessing=True).df

    testset_standardized = standardize_test_data(testset, dataset.column_means, dataset.column_stds)

    # Assuming the last column is the label
    X_test = testset_standardized.iloc[:, :-1].values
    y_test = testset.iloc[:, -1].values

    print("KNN score: ", knn.score(X_test, y_test))
    print("KMeans score: ", kmeans.score(X_test, y_test))