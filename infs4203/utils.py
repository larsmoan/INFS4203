import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd


# Note: you need a .env file in the root of your repository with REPO_DATA_DIR: "path" for this method to work
def get_data_dir():
    load_dotenv()
    dir_name = "REPO_DATA_DIR"
    if dir_name in os.environ:
        path = os.getenv(dir_name)
    else:
        return Exception(
            "No data directory found in the .env file or missing a .env file in the repo's root"
        )
    p = Path(path)
    return p


if __name__ == "__main__":
    print("Repo data directory: ", get_data_dir())


def standardize_test_data(test_df, column_means, column_stds):
    standardized_df = test_df.copy()
    for col in column_means.keys():
        standardized_df[col] = (test_df[col] - column_means[col]) / column_stds[col]
    return standardized_df

