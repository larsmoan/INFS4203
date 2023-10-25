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
from sklearn.metrics import f1_score
from utils import get_data_dir
import os

def standardize_test_data(test_df, column_means, column_stds):
    standardized_df = test_df.copy()
    for col in column_means.keys():
        standardized_df[col] = (test_df[col] - column_means[col]) / column_stds[col]
    return standardized_df

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


    
    # ----------------- Validating the algorithms on completely unseen data --------------
    validationset = INFS4203Dataset(
        "train2.csv", preprocessing=True
    ).df  # This will utilize the same class by imputing the NaN's present in the other train set
    # for the actual test set imputation will not be done. Just the standardization as done below.
    validationset_standardized = standardize_test_data(
        validationset, train_data.column_means, train_data.column_stds
    )

    # Assuming the last column is the label
    X_val = validationset_standardized.iloc[:, :-1].values
    X_val_not_std = validationset.iloc[:, :-1].values
    y_val = validationset.iloc[:, -1].values

    print("KNN score: ", knn.score(X_val, y_val))
    print("KMeans score: ", kmeans.score(X_val, y_val))
    print("Random Forest score: ", rf.score(X_val_not_std, y_val))


    #Using the random forest algorithm to predict the labels for the test set
    testset = pd.read_csv(get_data_dir() / "test.csv")
    testset_standardized = standardize_test_data(testset, train_data.column_means, train_data.column_stds)
    testset_not_std = testset.iloc[:, :].values

    y_pred = rf.predict(testset_not_std)
    
    #Generate report
    """ acc = rf.model.score(X_val_not_std, y_val)
    f1 = f1_score(y_val, rf.predict(X_val_not_std), average="macro")

    outputFormatter(y_pred, acc, f1, get_data_dir() / "s4827064.csv") """