from collections import Counter
from typing import List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import INFS4203Dataset
from models import (
    KMeansClassifier,
    KNNClassifier,
    RandomForestCLassifier,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils import get_data_dir, standardize_test_data
import os

def majority_voting(models: List, not_std_models: List, X: np.ndarray, X_not_std: np.ndarray):
    num_entries = len(X)
    predictions = []
    # Preallocate an array for predictions
    all_preds = np.zeros((num_entries, len(models)+len(not_std_models)))

    for i in range(len(models)):
        all_preds[:, i] = models[i].predict(X)
    for j in range(len(models), len(models) + len(not_std_models)):
        all_preds[:, j] = not_std_models[j-len(models)].predict(X_not_std)


    

    for i in range(num_entries):
        counter = Counter(all_preds[i])
        most_common_value = counter.most_common(1)[0][0]
        predictions.append(most_common_value)
    return predictions


# Code copied from the ed-discussion board for formatting the results report
def outputFormatter(pred, acc, f1, filename):
    # round acc and f1 to 3rd decimal place
    acc = "{:.3f}".format(acc)
    f1 = "{:.3f}".format(f1)
    if isinstance(pred, pd.DataFrame):
        pred = pred.values.tolist()
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    assert isinstance(
        pred, list
    ), "Unsupported type for pred. It should be either a list, numpy array or pandas dataframe"
    assert len(pred) == 300, "pred should be a list of 300 elements"
    pred_int = [int(x) for x in pred]
    csv_string = ",\n".join(map(str, pred_int))
    csv_string += ",\n" + acc + "," + f1
    filename = filename if filename.endswith(".csv") else filename + ".csv"
    with open(filename, "w") as f:
        f.write(csv_string)
    return csv_string


if __name__ == "__main__":
    # ---------- Load the dataset --------------------
    train_data = INFS4203Dataset("train.csv", preprocessing=True)
    #train_data.plotTSNE(train_data.df, plot_result=True, block=True)  # This is not part of the curriculum, but added to show the possibility of dimensionality reduction -> 2D.

    X = train_data.std_normalized_df[train_data.feature_columns].values
    X_not_std = train_data.cleaned_df[train_data.feature_columns].values

    y = train_data.std_normalized_df.Label.values   #Labels, common for both set of X_

    # ---------- Hyperparameters for the different models --------------------
    hyperparams_knn = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 300],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    hyperparams_kmeans = {
        "n_clusters": [3, 5, 7, 9, 11, 12, 13, 14, 15],
    }

    hyperparams_rf = {
        "n_estimators": np.linspace(3, 100, num=10, dtype=int),
        "max_depth": [1, 5, 20, 100],  # Maximum number of levels in tree
        "min_samples_split": [1, 5, 10, 30],
        "min_samples_leaf": [1, 3, 4],
        "criterion": ["gini", "entropy"],  # Criterion
    }

    # Run GridSearch CV on hyperparams, and fit the estimators to the train data.
    # ---------- KNN --------------------
    knn = KNNClassifier()
    knn.fit(X, y, hyperparams_knn)

    # ---------- KMeans --------------------
    kmeans = KMeansClassifier()
    kmeans.fit(X, y, hyperparams_kmeans)


    # ---------- Random Forest --------------------
    if os.path.isfile(get_data_dir() / "rf_best_estimator.joblib"):
        print("Have already fitted RF algo")
        rf = joblib.load(get_data_dir() / "rf_best_estimator.joblib")
    else:
        #We dont have a "fitted" random forest yet
        rf = RandomForestCLassifier()

        rf.fit(X_not_std, y, hyperparams_rf)  #Not using standardized data for X since this is not needed for RF
        joblib.dump(rf, get_data_dir() / "rf_best_estimator.joblib")    #Saving the optimized rf


    
    # ----------------- Testing the algorithms on completely unseen data --------------
    testset = INFS4203Dataset(
        "train2.csv", preprocessing=True
    ).df  # This will utilize the same class by imputing the NaN's present in the other train set
    # for the actual test set imputation will not be done. Just the standardization as done below.
    testset_standardized = standardize_test_data(
        testset, train_data.column_means, train_data.column_stds
    )

    # Assuming the last column is the label
    X_test = testset_standardized.iloc[:, :-1].values
    X_test_not_std = testset.iloc[:, :-1].values
    y_test = testset.iloc[:, -1].values

    print("KNN score: ", knn.score(X_test, y_test))
    print("KMeans score: ", kmeans.score(X_test, y_test))
    print("Random Forest score: ", rf.score(X_test_not_std, y_test))


    # ----------------- Majority voting --------------
    models = [knn, kmeans]
    not_std_models = [rf, rf, rf]
    predictions = majority_voting(models, not_std_models, X_test, X_test_not_std)

    print("Majority voting score: ", f1_score(y_test, predictions, average="macro"))
