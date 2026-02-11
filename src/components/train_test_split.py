"""Time-series train/validation/test split for SalesForecast."""
import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception.exception import SalesForecastException
from src.logging.logger import logging


@dataclass
class DataSplitConfig:
    data_path: str
    root_dir: str
    target_column: str = "MONTHLY_SALES"
    date_column: str = "DATE"


class DataSplit:
    def __init__(self, config: DataSplitConfig):
        self.config = config

    def split(self):
        try:
            full_df = pd.read_csv(self.config.data_path)

            if self.config.date_column not in full_df.columns:
                raise ValueError("DATE column is required for time-based split")

            unique_dates = sorted(list(full_df[self.config.date_column].unique()))
            if len(unique_dates) < 3:
                logging.warning(
                    "Less than 3 unique dates found. Falling back to minimal split."
                )

            if len(unique_dates) >= 3:
                train_index = unique_dates[:-2]
                valida_index = [unique_dates[-2]]
                test_index = [unique_dates[-1]]
            elif len(unique_dates) == 2:
                train_index = [unique_dates[0]]
                valida_index = [unique_dates[1]]
                test_index = [unique_dates[1]]
            else:
                train_index = unique_dates
                valida_index = unique_dates
                test_index = unique_dates

            drop_cols = [self.config.target_column, self.config.date_column]
            if "UNIQUE_ID" in full_df.columns:
                drop_cols.append("UNIQUE_ID")

            X_train = full_df[full_df[self.config.date_column].isin(train_index)].drop(
                drop_cols, axis=1
            )
            Y_train = full_df[full_df[self.config.date_column].isin(train_index)][
                self.config.target_column
            ]

            X_valida = full_df[full_df[self.config.date_column].isin(valida_index)].drop(
                drop_cols, axis=1
            )
            Y_valida = full_df[full_df[self.config.date_column].isin(valida_index)][
                self.config.target_column
            ]

            X_test = full_df[full_df[self.config.date_column].isin(test_index)].drop(
                drop_cols, axis=1
            )
            Y_test = full_df[full_df[self.config.date_column].isin(test_index)][
                self.config.target_column
            ]

            os.makedirs(self.config.root_dir, exist_ok=True)

            X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
            X_valida.to_csv(os.path.join(self.config.root_dir, "X_valida.csv"), index=False)
            Y_train.to_csv(os.path.join(self.config.root_dir, "Y_train.csv"), index=False)
            Y_test.to_csv(os.path.join(self.config.root_dir, "Y_test.csv"), index=False)
            Y_valida.to_csv(os.path.join(self.config.root_dir, "Y_valida.csv"), index=False)

            logging.info("Time series data split completed")
            print("Time series data split completed")
            return X_train, X_test, X_valida, Y_train, Y_test, Y_valida
        except Exception as e:
            raise SalesForecastException(e, sys)


if __name__ == "__main__":
    config = DataSplitConfig(
        data_path="artifacts/preprocessed/preprocessed.csv",
        root_dir="artifacts/split",
        target_column="MONTHLY_SALES",
        date_column="DATE",
    )

    splitter = DataSplit(config)
    splitter.split()
