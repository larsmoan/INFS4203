from collections import Counter
from typing import Any, Dict, List
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np

class KNNClassifier:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def fit(self, X, y, hyperparams: Dict[str, List[Any]]):
        self.hyperparams = hyperparams
        grid_search = GridSearchCV(
            self.model,
            self.hyperparams,
            cv=10,
            verbose=1,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print("KNN best params: ", grid_search.best_params_)
    
    def predict(self, X):   #Ugly, but needed for majority voting
        return self.model.predict(X)

    def score(self, X, y) -> tuple:
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: KNN classifier \n", report)
        
        f1_macro_scorer = make_scorer(f1_score, average='macro')
        f1_cv_scores = cross_val_score(self.model, X, y, cv=10, scoring=f1_macro_scorer)
        acc_cv_scores = cross_val_score(self.model, X, y, cv=10, scoring='accuracy')

        print("F1 avg 10cv score: ", np.mean(f1_cv_scores))
        print("Acc avg 10cv score:", np.mean(acc_cv_scores))
        return (np.mean(f1_cv_scores), np.mean(acc_cv_scores))

class KMeansClassifier:
    def __init__(self):
        self.model = KMeans(n_init=1)
        self.cluster_to_label_mapping = None

    def fit(self, X, y, hyperparams: Dict[str, List[Any]]):
        # Perform the grid search
        self.hyperparams = hyperparams
        grid_search = GridSearchCV(
            self.model, 
            self.hyperparams, 
            cv=10, 
            verbose=1,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid_search.fit(X,y)
        print("KMeans best params: ", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        cluster_assignments = self.model.labels_

        # Create a mapping from cluster ID to most frequent y value
        self.cluster_to_label_mapping = {}
        for cluster_id in set(cluster_assignments):
            true_labels_in_cluster = [int(y[i]) for i, label in enumerate(cluster_assignments) if label == cluster_id]
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            self.cluster_to_label_mapping[cluster_id] = most_common_label

    def predict(self, X):
        clusters = self.model.predict(X)
        return [self.cluster_to_label_mapping[cluster] for cluster in clusters]

    def score(self, X, y) -> tuple:
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: KMeans classifier \n", report)
        

        f1_macro_scorer = make_scorer(f1_score, average='macro')
        f1_cv_scores = cross_val_score(self.model, X, y, cv=10, scoring=f1_macro_scorer)
        acc_cv_scores = cross_val_score(self.model, X, y, cv=10, scoring='accuracy')


        print("F1 avg 10cv score: ", np.mean(f1_cv_scores))
        print("Acc avg 10cv score:", np.mean(acc_cv_scores))
        return (np.mean(f1_cv_scores), np.mean(acc_cv_scores))

class RandomForestCLassifier:
    def __init__(self):
        self.model = SklearnRandomForestClassifier()

    def fit(self, X, y, hyperparams: Dict[str, List[Any]]):
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

    
    def score(self, X, y) -> tuple:
        predicted_labels = self.predict(X)
        report = classification_report(y, predicted_labels)
        print("Classification report for: RandomForest classifier \n", report)
        

        f1_macro_scorer = make_scorer(f1_score, average='macro')
        f1_cv_scores = cross_val_score(self.model, X, y, cv=10, scoring=f1_macro_scorer)
        acc_cv_scores = cross_val_score(self.model, X, y, cv=10, scoring='accuracy')


        print("F1 avg 10cv score: ", np.mean(f1_cv_scores))
        print("Acc avg 10cv score:", np.mean(acc_cv_scores))
        return (np.mean(f1_cv_scores), np.mean(acc_cv_scores))

    def predict(self, X):
        return self.model.predict(X)