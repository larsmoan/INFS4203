from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report



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
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: KNN classifier \n", report)
        return f1_score(y, predicted_labels, average="macro")
    
    def predict(self, X):
        return self.knn.predict(X)

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
        report = classification_report(y, predicted_labels)
        print("Classification report for: KMeans classifier \n", report)
        return f1_score(y, predicted_labels, average="macro")
    
class DecisionTreeClassifier:
    def __init__(self):
        #Unpack the hyperparams
        self.decision_tree = SklearnDecisionTreeClassifier()
 
    
    def fit(self, X,y):
        self.decision_tree.fit(X,y)
        
    
    def predict(self, X):
        return self.decision_tree.predict(X)
    
    def score(self, X, y):
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: DecisionTree classifier \n", report)
        return f1_score(y, predicted_labels, average="macro")
