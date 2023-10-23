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
from collections import Counter
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt




from models import KNNClassifier, KMeansClassifier, DecisionTreeClassifier


def majority_voting(models: List, X: List):
    num_entries = len(X)
    predictions = []

    # Preallocate an array for predictions
    all_preds = np.zeros((num_entries, len(models)))

    for i in range(len(models)):
        all_preds[:, i] = models[i].predict(X)

    for i in range(num_entries):
        counter = Counter(all_preds[i])
        most_common_value = counter.most_common(1)[0][0]
        predictions.append(most_common_value)
    
    return predictions




if __name__ == '__main__':
    # ---------- Load the dataset --------------------
    train_data = INFS4203Dataset("train.csv", preprocessing=True)
    data = train_data.std_normalized_df
    X = data[train_data.feature_columns].values
    X_not_norm = train_data.cleaned_df[train_data.feature_columns].values   #Removed outliers but not normalized
    y = data.Label.values

    hyperparams_knn = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    hyperparams_kmeans = {
        "n_clusters": [3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18],
    }

    #Run GridSearch CV on hyperparams, and fit the estimators to the train data.
    # ---------- KNN --------------------
    knn = KNNClassifier(hyperparams_knn)
    knn.fit(X, y)
 
    # ---------- KMeans --------------------
    kmeans = KMeansClassifier(hyperparams_kmeans)
    kmeans.fit(X, y)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_not_norm, y)

    # ----------------- Testing the algorithms on completely unseen data --------------
    testset = INFS4203Dataset("train2.csv", preprocessing=True).df  #This will utilize the same class by imputing the NaN's present in the other train set
                                                                    # for the actual test set imputation will not be done. Just the standardization as done below.
    testset_standardized = standardize_test_data(testset, train_data.column_means, train_data.column_stds)
    

    # Assuming the last column is the label
    X_test = testset_standardized.iloc[:, :-1].values
    y_test = testset.iloc[:, -1].values

    print("KNN score: ", knn.score(X_test, y_test))
    print("KMeans score: ", kmeans.score(X_test, y_test))
    print("DTree score: ", dtree.score(X_test, y_test))

    # ----------------- Majority voting --------------
    models = [knn, kmeans, dtree]
    maj_predictions = majority_voting(models, X_test)
    print("Majority voting score: ", f1_score(y_test, maj_predictions, average="macro"))

    plot_tree(dtree.decision_tree, max_depth=3)
    plt.show()