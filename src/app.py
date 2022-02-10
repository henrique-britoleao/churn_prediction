###############################################################################
#####  App
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import constants as cst
import pickle
import shap
import seaborn as sns
from datetime import timedelta


st.set_option("deprecation.showPyplotGlobalUse", False)
sns.set_theme()
###############################################################################
DATASET = pd.read_csv(cst.FULL_PREDICTIONS_PATH, index_col=0)
with open(cst.EXPLAINER_PATH, "rb") as explainer_file:
    EXPLAINER = pickle.load(explainer_file)
with open(cst.PIPELINE_PATH, "rb") as explainer_file:
    PIPELINE = pickle.load(explainer_file)
SHAP_VALUES = np.load(cst.SHAP_VALUES_PATH)


class App:
    """Class to generate a streamlit app displaying the results of churn prediction."""

    FEATURE_NAMES = [
        "Avg qty",
        "Avg sales",
        "# branches",
        "# products",
        "Frequency",
        "# purchases",
        "Max qty",
        "Min qty",
        "Std qty",
        "Last order qty",
        "2nd last order qty",
        "3rd last order qty",
        "4th last order qty",
        "Max sales",
        "Min sales",
        "Std sales",
        "Last order sales",
        "2nd last order sales",
        "3rd last order sales",
        "4th last order sales",
        "Delay last and 2nd last purchase",
        "Delay 2nd last and 3rd last purchase",
        "Delay 3rd last and 4th last purchase",
        "Delay 4th last and 5th last purchase",
    ]
    MODEL_FEATURES = [
        "client_id",
        "mean_qty",
        "max_qty",
        "min_qty",
        "std_qty",
        "last_qty_1",
        "last_qty_2",
        "last_qty_3",
        "last_qty_4",
        "mean_sales",
        "max_sales",
        "min_sales",
        "std_sales",
        "last_sales_1",
        "last_sales_2",
        "last_sales_3",
        "last_sales_4",
        "n_branch",
        "n_product",
        "purchase_freq",
        "n_purchases",
        "delay_purchase_n1",
        "delay_purchase_n2",
        "delay_purchase_n3",
        "delay_purchase_n4",
        "client_age",
        "time_from_last_purchase",
        "client_lifetime",
        "last_purchase",
        "frequency",
        "client_category",
    ]

    def __init__(self, dataset, pipeline, shap_explainer, shap_vals):
        # self.dataset = dataset
        self.option = None
        self.dataset = dataset
        self.pipeline = pipeline
        self.shap_explainer = shap_explainer
        self.shap_vals = shap_vals

        (
            self.mean_sales,
            self.mean_qty,
            self.mean_purchases,
            self.mean_freq,
        ) = self._calculate_stats()

    def configure_page(self):
        """
        Configures app page
        Creates sidebar with selectbox leading to different main pages
        Returns:
            option (str): Name of main page selected by user
        """
        # create sidebar
        st.sidebar.title("Churn prediction")
        option = st.sidebar.selectbox(
            "Pick Dashboard:",
            (
                "Clients Overview",
                "Churn Analysis",
            ),
        )
        # client = st.sidebar.selectbox("Client ID", (self.dataset["client_id"].unique()))
        client = int(st.sidebar.text_input("Client ID", 14))

        self.option = option
        self.client = client

    def create_main_pages(self):
        """
        Creates pages for all options available in sidebar
        Page is loaded according to option picked in sidebar
        """
        if self.option == "Clients Overview":
            st.title("ClientCo Churn Analysis")

            self._display_last_sales()
            self._display_actionable_clients()
            self._display_churn()

        if self.option == "Churn Analysis":
            st.title("ClientCo Churn Analysis")
            st.header(f"Churn analysis (client ID:{self.client})")
            st.subheader(
                f"Churn probability: {self.dataset.loc[self.dataset['client_id']==self.client, 'churn_prob'].values[0]*100:.1f} %"
            )
            self._display_stats()
            self._display_force_plot()
            self._display_last_transactions()

    def _calculate_stats(self):
        return (
            self.dataset["mean_sales"].mean(),
            self.dataset["mean_qty"].mean(),
            self.dataset["n_purchases"].mean(),
            self.dataset["purchase_freq"].mean(),
        )

    def _display_actionable_clients(self):
        st.header("Actionable clients")
        actionable_clients = (
            self.dataset.loc[
                self.dataset["actionable"] == 1,
                [
                    "client_id",
                    "churn_prob",
                    "mean_sales",
                    "mean_qty",
                    "n_purchases",
                ],
            ]
            .sort_values(by="churn_prob", ascending=False)
            .set_index("client_id")
            .T
        )
        st.dataframe(actionable_clients)

    def _display_force_plot(self):
        st.subheader("Force plot")
        index = np.where(self.dataset["client_id"] == self.client)
        fig = shap.force_plot(
            self.shap_explainer.expected_value,
            self.shap_vals[index],
            features=self.pipeline.steps[0][1].fit_transform(
                self.dataset[App.MODEL_FEATURES]
            )[index],
            feature_names=App.FEATURE_NAMES,
            # link="logit",
            matplotlib=True,
        )
        st.pyplot(fig)

    def _display_last_transactions(self):
        st.subheader("Purchases in the last 90 days")
        # load last transactions
        last_transactions = _load_last_transactions()
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.title("Total purchases in last 90 days")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        ax.set_xlim(cst.MAX_DATE - timedelta(90), cst.MAX_DATE)

        sns.lineplot(
            data=last_transactions[last_transactions["client_id"] == self.client],
            x="date_order",
            y="sales_net",
            ax=ax,
            estimator="sum",
            ci=None,
        )
        for item in ax.get_xticklabels():
            item.set_rotation(45)

        st.pyplot(fig)

    def _display_stats(self):
        st.subheader("Client statistics")
        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        stats_df = pd.DataFrame(
            {
                "Purchase Frequence": [
                    self.dataset.loc[
                        self.dataset["client_id"] == self.client, "purchase_freq"
                    ].values[0]
                ],
                "Mean Order Quantity": [
                    self.dataset.loc[
                        self.dataset["client_id"] == self.client, "mean_sales"
                    ].values[0]
                ],
                "Mean Order Sales": [
                    self.dataset.loc[
                        self.dataset["client_id"] == self.client, "mean_qty"
                    ].values[0],
                ],
                "# Purchases": [
                    self.dataset.loc[
                        self.dataset["client_id"] == self.client, "n_purchases"
                    ].values[0],
                ],
            }
        )
        print(stats_df)
        sns.barplot(data=stats_df, y="Purchase Frequence", ax=ax[0], alpha=0.5)
        sns.barplot(
            data=stats_df, y="Mean Order Quantity", ax=ax[1], alpha=0.5, color="violet"
        )
        sns.barplot(
            data=stats_df, y="Mean Order Sales", ax=ax[2], color="red", alpha=0.5
        )
        sns.barplot(data=stats_df, y="# Purchases", ax=ax[3], color="green", alpha=0.5)
        ax[0].axhline(
            self.mean_freq, xmin=0.1, xmax=0.9, dashes=[1, 1], label="Avg Freq"
        )
        ax[1].axhline(
            self.mean_qty,
            xmin=0.1,
            xmax=0.9,
            dashes=[1, 1],
            label="Avg MOQ",
            color="violet",
        )
        ax[2].axhline(
            self.mean_sales,
            xmin=0.1,
            xmax=0.9,
            dashes=[1, 1],
            label="Avg MOS",
            color="red",
        )
        ax[3].axhline(
            self.mean_purchases,
            xmin=0.1,
            xmax=0.9,
            dashes=[1, 1],
            label="Avg # purchases",
            color="green",
        )
        plt.tight_layout()
        fig.legend()

        st.pyplot(fig)

    def _display_last_sales(self):
        st.title("Sales by category")
        # load last transactions
        last_transactions = _load_last_transactions()
        labels = _load_labels()
        transactions_labels = pd.merge(last_transactions, labels, how="left")
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.title("Total sales by client category")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        sns.lineplot(
            data=transactions_labels,
            x="date_order",
            y="sales_net",
            hue="client_category",
            ci=None,
            ax=ax,
            estimator="sum",
        )
        # for item in ax.get_xticklabels():
        #     item.set_rotation(45)

        st.pyplot(fig)

    def _display_churn(self):
        st.subheader("Churn by client category")
        churn_table = _load_churn_table()

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # plt.suptitle("Churn ratio")
        ax[0].title.set_text("Freq buyers")
        ax[0].pie(
            churn_table.loc[
                churn_table["client_category"] == "freq_buyer", "client_id"
            ],
            labels=["No churn", "Churn"],
            autopct="%.0f%%",
        )
        ax[1].title.set_text("Med-freq buyers")
        ax[1].pie(
            churn_table.loc[churn_table["client_category"] == "med_buyer", "client_id"],
            labels=["No churn", "Churn"],
            autopct="%.0f%%",
        )

        st.pyplot(fig)


@st.cache()
def _load_last_transactions():
    last_transactions = pd.read_csv(cst.VALIDATION_DATA_PATH, index_col=0)
    last_transactions["date_order"] = pd.to_datetime(last_transactions["date_order"])

    return last_transactions


@st.cache()
def _load_labels():
    return pd.read_csv(cst.TRAIN_TARGET_PATH, index_col=0)


@st.cache()
def _load_churn_table(dataset=DATASET):
    return dataset.groupby(by=["client_category", "is_churn"], as_index=False).agg(
        {
            "client_id": "count",
        }
    )


if __name__ == "__main__":
    app = App(DATASET, PIPELINE, EXPLAINER, SHAP_VALUES)
    app.configure_page()
    app.create_main_pages()
