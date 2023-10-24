import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_data_dir


class INFS4203Dataset:
    def __init__(self, csv_file, preprocessing=True):
        self.df = pd.read_csv(get_data_dir() / csv_file)

        self.cat_cols = self.df.columns[100:128]
        self.numerical_columns = self.df.columns[:100]
        self.feature_columns = self.df.columns[:128]
        self.labels = [
            "airplane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.W_values_dict = {}  # Initializing the dictionary to store W values

        self.std_scaler = StandardScaler()
        self.column_means = {}  # to store means of columns
        self.column_stds = {}  # to store standard deviations of columns

        if preprocessing:
            self.df = self.impute_values()  # Imputes the NaN's
            self.cleaned_df, self.anomaly_records = self.anomaly_detection(
                n_std_dev=4
            )  # Removes the outliers
            (
                self.min_max_normalized_df,
                self.std_normalized_df,
            ) = self.normalize()  # Normalizes the cleaned_df

    def __getitem__(self, idx):
        # Return a sample from the dataframe with corresponding label
        data = self.df.iloc[idx]
        features = data[:128].values.astype("float32")
        label = data[128:].values.astype("int32")
        return features, label

    # Does the dimensionality reduction of the dataset from 129 components -> 2 and saves them as their own columns
    def plotTSNE(self, df, plot_result=True, block=True):
        df_tmp = df.copy()
        df_numeric = df_tmp[self.feature_columns]

        m = TSNE(learning_rate=50)
        tsne_features = m.fit_transform(df_numeric)
        df_tmp["x_tsne"] = tsne_features[:, 0]
        df_tmp["y_tsne"] = tsne_features[:, 1]

        if plot_result:
            self.scatter_plot(
                "x_tsne", "y_tsne", df_tmp, "tSNE dimensionality reduction -> 2D"
            )

    def scatter_plot(self, col_1, col_2, df, title):
        # Plots the two colums against eachother for all the classes.
        plt.figure()
        sns.scatterplot(
            x=col_1,
            y=col_2,
            hue="Label",
            data=df,
            legend="full",
            hue_norm=(0, 10),
            palette="Set1",
        ).set_title(title)
        plt.legend(prop={"size": 8})
        plt.show()

    def plot_distribution(self, column_name, df, block):
        subframes = [df[df["Label"] == i] for i in range(0, 10)]
        num_cols = 5
        num_rows = len(subframes) // num_cols + (len(subframes) % num_cols > 0)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

        for i, subframe in enumerate(subframes):
            row_idx = i // num_cols
            col_idx = i % num_cols

            subframe.hist(column=column_name, bins=60, ax=axes[row_idx, col_idx])
            axes[row_idx, col_idx].set_title(
                f"Distribution of {column_name} : {self.labels[int(subframe['Label'].iloc[0])]}"
            )

        for i in range(len(subframes), num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            fig.delaxes(axes[row_idx, col_idx])

        plt.tight_layout()
        plt.show(block=block)

    def shapiro_wilk(self):
        subframes = [self.df[self.df["Label"] == i] for i in range(0, 10)]
        num_columns = len(self.df.columns) - 1  # Excluding 'Label' column

        # Create a figure and axes for 2x5 subplots
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))

        for i, subframe in enumerate(subframes):
            ax = axes[i // 5][i % 5]  # Determine the current axis
            W_values = []

            for column in subframe.columns:
                # Exclude the 'Label' column from the test
                if column != "Label":
                    W, p = stats.shapiro(subframe[column])
                    W_values.append(W)

            # Plotting on the current subplot axis (ax)
            ax.bar(range(num_columns), W_values)
            ax.set_ylim(0, 1.1)  # Since W values range between 0 and 1
            ax.set_ylabel("W Value")
            ax.set_title(f"W Values for Label {i}")

            self.W_values_dict[i] = W_values

        fig.suptitle("Shapiro-Wilk Test W Values for All Labels")  # Add main title
        plt.tight_layout()
        plt.show()

    def anomaly_detection(self, n_std_dev=3) -> pd.DataFrame:
        # Segment the dataframe based on unique values in the 'Label' column
        subframes = [self.df[self.df["Label"] == i] for i in range(0, 10)]
        cleaned_subframes = []  # List to store subframes after removing outliers
        anomalies_records = []  # List to store details about anomalies

        for i, subframe in enumerate(subframes):
            # Get boolean mask for rows to keep
            is_within_bounds = np.ones(len(subframe), dtype=bool)

            for column in self.numerical_columns:
                # Compute mean and standard deviation for the column
                mean, std = subframe[column].mean(), subframe[column].std()

                # Get boolean mask for outliers in the column
                is_outlier = (subframe[column] < mean - n_std_dev * std) | (
                    subframe[column] > mean + n_std_dev * std
                )

                # Update the mask for rows to keep
                is_within_bounds &= ~is_outlier

                # Record columns and classes with anomalies
                if is_outlier.sum() > 0:
                    anomalies_records.append(
                        {"Label": i, "Column": column, "Count": is_outlier.sum()}
                    )

            # Keep only the rows without outliers
            cleaned_subframe = subframe[is_within_bounds]
            cleaned_subframes.append(cleaned_subframe)

        # Concatenate the cleaned subframes to get the final cleaned dataframe
        cleaned_df = pd.concat(cleaned_subframes, axis=0).reset_index(drop=True)

        # Return cleaned dataframe and anomaly records
        return cleaned_df, anomalies_records

    # Used to impute both numerical and categorical values for NaN values present in the original dataset
    def impute_values(self) -> pd.DataFrame:
        unique_label = self.df["Label"].unique()
        subframes = [self.df[self.df["Label"] == elem] for elem in unique_label]

        resulting_subframes = []
        for subframe in subframes:
            cat_mode = (
                subframe[self.cat_cols].mode().iloc[0]
            )  # Most common value used to impute the cateogrical columns
            num_mean = subframe[
                self.numerical_columns
            ].mean()  # Mean for the numerical columns
            subframe = subframe.fillna(num_mean)
            subframe = subframe.fillna(cat_mode)

            resulting_subframes.append(subframe)

        df_tmp = pd.concat(resulting_subframes)
        df_tmp.reset_index(inplace=True, drop=True)

        return df_tmp

    def normalize(self) -> pd.DataFrame:
        columns_to_normalize = self.feature_columns
        min_max_scaler = MinMaxScaler()

        df_std_normalized = self.cleaned_df.copy()
        df_min_max_normalized = self.cleaned_df.copy()

        df_min_max_normalized[columns_to_normalize] = min_max_scaler.fit_transform(
            df_min_max_normalized[columns_to_normalize]
        )

        df_std_normalized[columns_to_normalize] = self.std_scaler.fit_transform(
            df_std_normalized[columns_to_normalize]
        )

        # Save means and stds for each column
        for col in columns_to_normalize:
            self.column_means[col] = self.std_scaler.mean_[
                list(columns_to_normalize).index(col)
            ]
            self.column_stds[col] = np.sqrt(self.std_scaler.var_)[
                list(columns_to_normalize).index(col)
            ]

        return df_std_normalized, df_min_max_normalized


if __name__ == "__main__":
    dset = INFS4203Dataset("train.csv", preprocessing=True)
