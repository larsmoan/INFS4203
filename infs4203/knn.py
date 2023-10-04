from dataset import INFS4203Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Load the dataset
dataset = INFS4203Dataset("train.csv", preprocessing=True)

X = dataset.min_max_normalized_df[dataset.feature_columns].values
print(X)
y = dataset.min_max_normalized_df.Label.values

labels = dataset.cleaned_df.Label
print(labels)

knn = KNeighborsClassifier(n_neighbors=5)

# Initialize StratifiedKFold for cross-validation with shuffling
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

accuracies = []
f1_scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    # Print the distribution of the labels per fold
    print(pd.Series(predictions).value_counts())

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    accuracies.append(acc)
    f1_scores.append(f1)

    print("Accuracy:", acc)
    print("F1 Score:", f1)

print("Average Accuracy:", sum(accuracies) / len(accuracies))
print("Average F1 Score:", sum(f1_scores) / len(f1_scores))
