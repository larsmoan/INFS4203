from utils import get_data_dir
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib

class INFS4203Dataset():
    def __init__(self, csv_file, impute_nans=None, normalize=None, outlier_detection=None):
        self.df = pd.read_csv(get_data_dir() / csv_file)
        if len(self.df) < 10:
            print("Couldnt load df")
        if impute_nans:
            self.df = impute_nans(self.df)
        
        #Class member 
        self.non_numerical_cols = self.df.columns[100:]
        self.numerical_columns = self.df.columns[:100]
        self.labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #Return a sample from the dataframe with corresponding label
        data = self.df.iloc[idx]
        features = data[:128].values.astype("float32")
        label = data[128:].values.astype("int32")
        return features, label

    #Does the dimensionality reduction of the dataset from 129 components -> 2 and saves them as their own columns
    def addTSNE(self, plot_result=False):
        df_numeric = self.df.drop(self.non_numerical_cols, axis=1)

        m = TSNE(learning_rate=50)
        tsne_feattures = m.fit_transform(df_numeric)
        self.df['x_tsne'] = tsne_feattures[:, 0]
        self.df['y_tsne'] = tsne_feattures[:, 1]

        if plot_result:
            sns.scatterplot(x='x_tsne', y='y_tsne', hue='Label', data=self.df, legend='full', hue_norm=(0,10), palette='Set1').set_title("Dimensionality reduction TSNE")
            plt.legend(prop={'size': 8})  # Adjust the font size here
            plt.show()


    
    def plot_distribution(self, column_name):
        subframes = [self.df[self.df['Label'] == i] for i in range(0,10)]
        num_cols = 5

        num_rows = len(subframes) // num_cols + (len(subframes) % num_cols > 0)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4*num_rows))
        
        for i, subframe in enumerate(subframes):
            row_idx = i // num_cols
            col_idx = i % num_cols

            subframe.hist(column=column_name, bins=60, ax=axes[row_idx, col_idx])
            axes[row_idx, col_idx].set_title(f"Distribution of {column_name} : {self.labels[int(subframe['Label'].iloc[0])]}")

        for i in range(len(subframes), num_rows*num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            fig.delaxes(axes[row_idx, col_idx])
        plt.tight_layout()
        plt.show()
        

#Used to impute both numerical and categorical values for NaN values present in the original dataset
def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    unique_label = df['Label'].unique()
    subframes = [df[df['Label'] == elem] for elem in unique_label]    

    resulting_subframes = []
    for subframe in subframes:
        num_cols = subframe.columns[:100]   
        cat_cols = subframe.columns[100:128]

        cat_mode = subframe[cat_cols].mode().iloc[0]    #Most common value used to impute the cateogrical columns
        num_mean = subframe[num_cols].mean()            #Mean for the numerical columns
        subframe = subframe.fillna(num_mean)
        subframe = subframe.fillna(cat_mode)

        resulting_subframes.append(subframe)
    
    df_tmp = pd.concat(resulting_subframes)
    df_tmp.reset_index(inplace=True, drop=True)
    return df_tmp

#V1: I think this just brute forces the normalization. Need to do some outlier detection before this can be applied properly.
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_normalize = df.columns[:128]
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df



def outlier_detection(df: pd.DataFrame) -> pd.DataFrame:
    #Calculathe the z-score and plot the outliers in a different color
    z = np.abs(stats.zscore(df['Num_Col_0']))
    plt.plot(z)
    plt.show()



if __name__ == "__main__":
    dset = INFS4203Dataset('train.csv', impute_nans=impute_values)
    dset.addTSNE(plot_result=True)
    print(dset.df.head())
