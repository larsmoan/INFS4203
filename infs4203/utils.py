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


def print_terminal_histogram(y_test):
    # Get counts for each label, ensuring labels 0-9 are used
    counts = (
        pd.Series(y_test).value_counts(normalize=True).reindex(range(10), fill_value=0)
    )

    # Maximum length of the bar
    max_length = 40

    for index, value in counts.items():
        bar_length = int(value * max_length)
        bar = "■" * bar_length
        # Align bars to the right for consistent visualization
        print(f"{index}: {bar.rjust(max_length)} {value:.2f}")
