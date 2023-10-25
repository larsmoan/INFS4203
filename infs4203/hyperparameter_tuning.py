from dataset import INFS4203Dataset

from models import (
    KMeansClassifier,
    KNNClassifier,
    RandomForestCLassifier,
)
from sklearn.metrics import f1_score
from utils import get_data_dir
import numpy as np
import joblib


# Load the training data
train_data = INFS4203Dataset("train_strat_2.csv", preprocessing=True)

# Optional - plot the tSNE dimensionality reduction for inspecting the dataset
#train_data.plotTSNE(train_data.cleaned_df)
#train_data.plotTSNE(train_data.df)

X_train_stdnorm = train_data.std_normalized_df[train_data.feature_columns].values
X_train = train_data.cleaned_df[train_data.feature_columns].values

y_train = train_data.std_normalized_df.Label.values

hyperparams_knn = {
    "n_neighbors": [1,2, 3, 5, 7, 20, 40, 100, 300, 400],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}

hyperparams_kmeans = {
    "n_clusters": [10, 11, 12, 15, 17],
}

hyperparams_rf = {
    "n_estimators": np.linspace(3, 100, num=4, dtype=int),
    "max_depth": [3, 5, 6, 10],  # Maximum number of levels in tree
    "min_samples_split": [2, 5, 7, 10],
    "min_samples_leaf": [4, 5],
    "criterion": ["gini", "entropy"],  # Criterion
}

# ---------- KNN --------------------
knn = KNNClassifier()
knn.fit(X_train_stdnorm, y_train, hyperparams_knn)
knn.score(X_train_stdnorm, y_train)


# ---------- KMeans --------------------
kmeans = KMeansClassifier()
kmeans.fit(X_train_stdnorm, y_train, hyperparams_kmeans)
kmeans.score(X_train_stdnorm, y_train)

# ------- Random Forest -----------
rf = RandomForestCLassifier()
rf.fit(X_train, y_train, hyperparams_rf)
rf.score(X_train, y_train)

#Save the tuned models to data/ dir
joblib.dump(knn, get_data_dir() / "knn_best.joblib")
joblib.dump(kmeans, get_data_dir() / "kmeans_best.joblib")
joblib.dump(rf, get_data_dir() / "rf_best.joblib")