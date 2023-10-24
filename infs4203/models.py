from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.tree import plot_tree
from tqdm import tqdm


class KNNClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)

    def fit(self, X, y, hyperparams: Optional[Dict[str, List[Any]]] = None):
        if hyperparams:
            self.hyperparams = hyperparams
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
        else:
            self.knn.fit(X, y)
            print(
                "Fitted KNN with default hyperparams: a score of: ",
                self.knn.score(X, y),
                "was achieved",
            )

    def score(self, X, y):
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: KNN classifier \n", report)
        return f1_score(y, predicted_labels, average="macro")

    def predict(self, X):
        return self.knn.predict(X)


class KMeansClassifier:
    def __init__(self):
        self.kmeans = KMeans(n_init=1)
        self.cluster_to_label_mapping = None

    def fit(self, X, y, hyperparams: Dict[str, List[Any]]):
        # Perform the grid search
        self.hyperparams = hyperparams
        grid_search = GridSearchCV(
            self.kmeans, self.hyperparams, cv=10, verbose=1, n_jobs=-1
        )
        grid_search.fit(X)
        print("KMeans best params: ", grid_search.best_params_)
        self.kmeans = grid_search.best_estimator_
        cluster_assignments = self.kmeans.labels_

        # Create a mapping from cluster ID to most frequent y value
        self.cluster_to_label_mapping = {}
        for cluster_id in set(cluster_assignments):
            true_labels_in_cluster = [
                y[i]
                for i, label in enumerate(cluster_assignments)
                if label == cluster_id
            ]
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            self.cluster_to_label_mapping[cluster_id] = most_common_label

    def predict(self, X):
        clusters = self.kmeans.predict(X)
        return [self.cluster_to_label_mapping[cluster] for cluster in clusters]

    def score(self, X, y):
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: KMeans classifier \n", report)
        return f1_score(y, predicted_labels, average="macro")





class RandomForestCLassifier:
    def __init__(self):
        self.model = SklearnRandomForestClassifier()

    def fit(self, X, y, hyperparams: Optional[Dict[str, List[Any]]] = None):
        if hyperparams:
            self.hyperparams = hyperparams
            grid_search = GridSearchCV(
                self.model,
                self.hyperparams,
                cv=10,
                verbose=4,
                scoring="f1_macro",
                n_jobs=-1,
            )
            grid_search.fit(X, y)

            # Update the model to the best resulting estimator
            self.model = grid_search.best_estimator_

            print("RF best params: ", grid_search.best_params_)
            print("RF best score: ", grid_search.best_score_)
        else:
            self.model.fit(X, y)
            print(
                "Fitted RandomForest with default hyperparams: a score of: ",
                self.model.score(X, y),
                "was achieved",
            )
    

    def score(self, X, y):
        predicted_labels = self.model.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: RandomForest classifier \n", report)
        return f1_score(y, predicted_labels, average="macro")

    def predict(self, X):
        return self.model.predict(X)