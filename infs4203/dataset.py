from torch.utils.data import Dataset
from utils import get_data_dir
import pandas as pd


class INFS4203Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(get_data_dir() / csv_file)

    def __len__(self):
        return len(self.df)


dset = INFS4203Dataset("train.csv")
print(len(dset))
