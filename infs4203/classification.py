from dataset import INFS4203Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from utils import print_terminal_histogram



# HYPERPARAMETERS
n_neighbours = 3
n_random_columns = 50
k_splits = 30

dset = INFS4203Dataset('train.csv', preprocessing=True)
data = dset.cleaned_df
print(len(data))

y = data.iloc[:, -1].values

sampled_data = data.iloc[:, :-1].sample(n=n_random_columns, axis=1)
print("Selected columns:", sampled_data.columns.tolist())

if 'Label' in sampled_data.columns:
    print("Error, label is somehow a part of the feature set")
    exit()

X = sampled_data.values


kf = KFold(n_splits=k_splits, random_state=None, shuffle=True)

accuracies = []  # List to store accuracy for each fold

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    # Split the data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbours)
    neigh.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = neigh.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    #Print the distribution of classes 
    print(f"\nFold: {i} - Length of test set: {len(y_test)}")
    print("Class distribution:")
    print_terminal_histogram(y_test)


    accuracies.append(accuracy)
    print(f" Accuracy for fold: {i} = {accuracy:.4f}")

# Calculate the average accuracy over all folds
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy over {kf.get_n_splits()} folds: {avg_accuracy:.4f}")