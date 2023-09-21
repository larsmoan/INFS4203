from utils import get_data_dir
import pandas as pd

class INFS4203Dataset():
    def __init__(self, csv_file, transforms=None):
        self.df = pd.read_csv(get_data_dir() / csv_file)
        if transforms:
            for transform in transforms:
                self.df = transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #Return a sample from the dataframe with corresponding label
        data = self.df.iloc[idx]
        features = data[:128].values.astype("float32")
        label = data[128:].values.astype("int32")
        return features, label


#Used to impute both numerical and categorical values for NaN values present in the original dataset
def impute_values(df):
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
    
    return pd.concat(resulting_subframes)


dset = INFS4203Dataset("train.csv", [impute_values])
print(dset.df.head())