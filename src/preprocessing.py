###############################################################################
#####  Data Preprocesssing
###############################################################################
import pandas as pd
import constants as cst
from datetime import timedelta

#####  Get logger
import logging

logger = logging.getLogger(__name__)

#####  Preprocessor
class Preprocessor:
    def __init__(self) -> None:
        pass

    def preprocess(
        self,
        data: pd.DataFrame,
        train_val_split: bool = True,
        val_split_lenght: int = 90,
    ) -> pd.DataFrame:
        data = self.enforce_data_quality(data)
        if train_val_split:
            self.get_validation_set(data, lenght=val_split_lenght)

        raise NotImplementedError()

    def enforce_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        # remove zero/negative quantities
        data = data[data["quantity"] > 0]
        # remove zero/negative sales
        data = data[data["sales_net"] > 0]
        # enforce date types
        data["date_order"] = pd.to_datetime(data["date_order"])
        data["date_invoice"] = pd.to_datetime(data["date_invoice"])

        data = data[
            (data["date_order"].dt.year >= 2015)
            & (data["date_invoice"].dt.year >= 2015)
        ]

        logger.info("Done performing data quality checks")

        return data

    def get_validation_set(self, data: pd.DataFrame, lenght: int) -> None:
        split_day = cst.MAX_DATE - timedelta(lenght)

        train_set = data.loc[data["date_order"] < split_day]
        val_set = data.loc[data["date_order"] >= split_day]

        logger.info("Done splitting validation and train transactions")

        train_set.to_csv(cst.TRAIN_TRANSACTION_PATH)
        val_set.to_csv(cst.VALIDATION_DATA_PATH)

        logger.info("Done saving validation and train transactions")
