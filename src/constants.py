###############################################################################
#####  Constants
###############################################################################
from datetime import datetime
import os

# paths
DATA_PATH = "../data"
INPUT_DATA_PATH = os.path.join(DATA_PATH, "inputs", "transactions_dataset.csv")
VALIDATION_DATA_PATH = os.path.join(DATA_PATH, "validation", "val_transactions.csv")
TRAIN_TRANSACTION_PATH = os.path.join(DATA_PATH, "train", "train_transactions.csv")

# Data metadata
MAX_DATE = datetime.strptime("2019-09-22", "%Y-%m-%d")
MIN_DATE = datetime.strptime("2017-09-22", "%Y-%m-%d")
