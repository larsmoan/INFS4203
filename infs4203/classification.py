from collections import Counter
from typing import List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import INFS4203Dataset
from sklearn.metrics import f1_score
from utils import get_data_dir
from sklearn.model_selection import cross_val_score

def standardize_test_data(test_df, column_means, column_stds):
    standardized_df = test_df.copy()
    for col in column_means.keys():
        standardized_df[col] = (test_df[col] - column_means[col]) / column_stds[col]
    return standardized_df

def majority_voting(std_models: List, models: List, X_std: np.ndarray, X: np.ndarray):
    num_entries = len(X)
    predictions = []
    # Preallocate an array for predictions
    all_preds = np.zeros((num_entries, len(models)+len(std_models)))

    for i in range(len(std_models)):
        all_preds[:, i] = std_models[i].predict(X_std)

    for j in range(len(std_models), len(std_models) + len(models)):
        all_preds[:, j] = std_models[j-len(models)].predict(X)

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

    # ----------------- Validating the algorithms on completely unseen data --------------
    trainingset = INFS4203Dataset("train_strat.csv", preprocessing=True)
    column_means = trainingset.column_means
    column_stds = trainingset.column_stds

    #This yields the dataframe with imputed NaN values, outliers are not removed.
    validationdata = INFS4203Dataset("val_strat.csv", preprocessing=True).df
    X_val = validationdata.iloc[:, :-1].values
    y_val = validationdata.iloc[:, -1].values

    #Standardize the test data according to the means and std's computed on the training set
    X_val_std = standardize_test_data(validationdata, column_means, column_stds).iloc[:, :-1].values

    #Load the models from best checkpoints - will print the best hyperparameters provided from the search
    knn = joblib.load(get_data_dir() / "knn_best.joblib")
    print(knn.model)
    kmeans = joblib.load(get_data_dir() / "kmeans_best.joblib")
    print(kmeans.model)
    rf = joblib.load(get_data_dir() / "rf_best.joblib")
    print(rf.model)

    #Score the models on the validation data
    print("KNN score: ", knn.score(X_val_std, y_val))
    print("KMeans score: ", kmeans.score(X_val_std, y_val))
    print("Random Forest score: ", rf.score(X_val, y_val))

    #Majority voting test
    models = [rf, rf]
    std_models = [knn, kmeans]
    pred = majority_voting(std_models, models, X_val_std, X_val)
    majority_f1 = f1_score(y_val, pred, average='macro')
    print("Majority voting F1",  majority_f1)


    #Generating the report - using the random forest algorithm - NOT majority voting
    f1, acc = rf.score(X_val, y_val)

    #Using the random forest algorithm to predict the labels for the test set
    testset = pd.read_csv(get_data_dir() / "test.csv")    
    testset = testset.iloc[:, :].values #This is the featurespace used for the RF-algo on the test set

    y_pred = rf.predict(testset)
    outputFormatter(y_pred, acc, f1, "s4827064.csv")