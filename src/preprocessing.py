###############################################################################
#####  Data Preprocesssing
###############################################################################
from typing import Tuple
import pandas as pd
import constants as cst
from datetime import timedelta
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split

#####  Get logger
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

#####  Preprocessor
class Preprocessor:
    def __init__(self, val_split_length: int = 90) -> None:
        self.split_day = cst.MAX_DATE - timedelta(val_split_length)

    def preprocess(
        self,
        data: pd.DataFrame = None,
        train_val_split: bool = True,
    ) -> pd.DataFrame:
        if train_val_split:
            data = self.enforce_data_quality(data)
            self.get_validation_set(data)

        # load train_data
        data = pd.read_csv(cst.TRAIN_TRANSACTION_PATH, index_col=0)
        data["date_order"] = pd.to_datetime(data["date_order"])
        data = data.sort_values("date_order")

        data = self.feature_engineering(data)
        data.to_csv(cst.FEATURES_PATH)

        labels = self.get_labels(data)
        train_set, test_set = self.get_train_test_sets(
            pd.read_csv(cst.FEATURES_PATH, index_col=0), labels
        )

        return train_set, test_set

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

    def get_validation_set(self, data: pd.DataFrame) -> None:
        train_set = data.loc[data["date_order"] < self.split_day]
        val_set = data.loc[data["date_order"] >= self.split_day]

        logger.info("Done splitting validation and train transactions")

        train_set.to_csv(cst.TRAIN_TRANSACTION_PATH)
        val_set.to_csv(cst.VALIDATION_DATA_PATH)

        logger.info("Done saving validation and train transactions")

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Started feature engineering")

        features = data.groupby(by="client_id", as_index=False).agg(
            online_percent=("order_channel", partial(item_freq, item="online")),
            phone_percent=("order_channel", partial(item_freq, item="by phone")),
            store_percent=("order_channel", partial(item_freq, item="at the store")),
            visit_percent=(
                "order_channel",
                partial(item_freq, item="during the visit of a sales rep"),
            ),
            other_percent=("order_channel", partial(item_freq, item="other")),
            mean_qty=("quantity", "mean"),
            max_qty=("quantity", "max"),
            min_qty=("quantity", "min"),
            std_qty=("quantity", "std"),
            last_qty_1=("quantity", partial(last, n=1)),
            last_qty_2=("quantity", partial(last, n=2)),
            last_qty_3=("quantity", partial(last, n=3)),
            last_qty_4=("quantity", partial(last, n=4)),
            mean_sales=("sales_net", "mean"),
            max_sales=("sales_net", "max"),
            min_sales=("sales_net", "min"),
            std_sales=("sales_net", "std"),
            last_sales_1=("sales_net", partial(last, n=1)),
            last_sales_2=("sales_net", partial(last, n=2)),
            last_sales_3=("sales_net", partial(last, n=3)),
            last_sales_4=("sales_net", partial(last, n=4)),
            n_branch=("branch_id", "nunique"),
            n_product=("product_id", "nunique"),
            purchase_freq=("date_order", purchase_frequency),
            delay_purchase_n1=("date_order", partial(time_delay, n=1)),
            delay_purchase_n2=("date_order", partial(time_delay, n=2)),
            delay_purchase_n3=("date_order", partial(time_delay, n=3)),
            delay_purchase_n4=("date_order", partial(time_delay, n=4)),
            client_age=("date_order", partial(time_from_today, n=0)),
            time_from_last_purchase=("date_order", partial(time_from_today, n=-1)),
            client_lifetime=("date_order", lifetime),
        )

        logger.info("Done feature engineering")

        return features

    def get_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        # get purchasing frequencies
        freqs = data.groupby(by="client_id", as_index=False).agg(
            last_purchase=("date_order", "max"),
            frequency=("date_order", purchase_frequency),
        )

        # setting churn deffinitions
        freq_churn = self.split_day - timedelta(30)
        medium_churn = self.split_day - timedelta(90)

        # assiging client categories
        freqs.loc[
            (freqs["frequency"] > 0) & (freqs["frequency"] < 10), "client_category"
        ] = "freq_buyer"
        freqs.loc[
            (freqs["frequency"] >= 10) & (freqs["frequency"] < 32), "client_category"
        ] = "med_buyer"
        freqs.loc[
            (freqs["frequency"] >= 32) | (freqs["frequency"] == 0), "client_category"
        ] = "infreq_buyer"

        # assigning labels
        freqs.loc[
            (freqs["client_category"] == "freq_buyer")
            & (freqs["last_purchase"] <= freq_churn),
            "is_churn",
        ] = 1
        freqs.loc[
            (freqs["client_category"] == "freq_buyer")
            & (freqs["last_purchase"] > freq_churn),
            "is_churn",
        ] = 0
        freqs.loc[
            (freqs["client_category"] == "med_buyer")
            & (freqs["last_purchase"] <= medium_churn),
            "is_churn",
        ] = 1
        freqs.loc[
            (freqs["client_category"] == "med_buyer")
            & (freqs["last_purchase"] > medium_churn),
            "is_churn",
        ] = 0

        logger.info("Done getting labels")

        freqs.to_csv(cst.TRAIN_TARGET_PATH)

        return freqs

    def get_train_test_sets(
        self, training_data: pd.DataFrame, training_target: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_training_data = pd.merge(
            training_data, training_target, on="client_id", how="left"
        )
        full_training_data.dropna(inplace=True)
        train_set, test_set = train_test_split(
            full_training_data, test_size=0.3, random_state=42
        )

        logger.info("Done splitting train and test set")

        train_set.to_css(cst.TRAIN_SET_PATH)
        test_set.to_css(cst.TEST_SET_PATH)

        logger.info("Done saving train and test set")

        return train_set, test_set


def purchase_frequency(dates):
    return pd.Timedelta(np.diff(dates.unique()).mean()).total_seconds() / (60 * 60 * 24)


def time_delay(dates, n=1):
    try:
        return pd.Timedelta(dates.iloc[-n] - dates.iloc[-n - 1]).total_seconds() / (
            60 * 60 * 24
        )
    except IndexError:
        return np.nan


def lifetime(dates):
    return pd.Timedelta(dates.iloc[-1] - dates.iloc[0]).total_seconds() / (60 * 60 * 24)


def time_from_today(dates, n=0):
    return pd.Timedelta(cst.MAX_DATE - dates.iloc[n]).total_seconds() / (60 * 60 * 24)


def last(sequence, n=1):
    try:
        return sequence.iloc[-n]
    except IndexError:
        return np.nan


def item_freq(series, item):
    if not np.isin(item, series):
        return 0
    else:
        return series.value_counts(normalize=True)[item]
