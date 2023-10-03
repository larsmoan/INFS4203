from dataset import INFS4203Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dset = INFS4203Dataset('train.csv', preprocessing=True)

print(dset.normalized_df.tail())

kf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)

# Assuming the last column of the dataset is the target label
X = dset.normalized_df.iloc[:, :-1].values
y = dset.normalized_df.iloc[:, -1].values

accuracies = []  # List to store accuracy for each fold

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    # Split the data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = neigh.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f" Accuracy for fold: {i} = {accuracy:.4f}")

# Calculate the average accuracy over all folds
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy over {kf.get_n_splits()} folds: {avg_accuracy:.4f}")
