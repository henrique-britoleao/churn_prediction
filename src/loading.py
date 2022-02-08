###############################################################################
#####  Data Preprocesssing
###############################################################################
import pandas as pd
import constants as cst


class Loader:
    def __init__(self, sample=None) -> None:
        self.sample = sample

    def load_data(self, path=cst.INPUT_DATA_PATH):
        if self.sample:
            return pd.read_csv(path, sep=";", nrows=self.sample)
        else:
            return pd.read_csv(path, sep=";")
