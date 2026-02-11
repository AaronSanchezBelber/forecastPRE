"""Data transformation pipeline for SalesForecast."""
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.exception.exception import SalesForecastException
from src.logging.logger import logging


@dataclass
class DataTransformationConfig:
    root_dir: str
    data_path: str
    data_path_df: str | None = None
    data_path_df_monthly_agg: str | None = None


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_df(self, save: bool = True) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.config.data_path,
                sep=",",
                quotechar='"',
                escapechar="\\",
                engine="python",
                on_bad_lines="warn",
            )
            df.columns = map(str.upper, df.columns)
            df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
            df["HOUR"] = df["DATE"].dt.hour
            df["DAY_OF_WEEK"] = df["DATE"].dt.day_of_week
            df["CITY_ID"] = OrdinalEncoder().fit_transform(df[["CITY"]])
            df.rename(columns={"CITY": "CITY_NAME", "ITEM_CNT_DAY": "SALES"}, inplace=True)
            if save:
                os.makedirs(self.config.root_dir, exist_ok=True)
                out = os.path.join(self.config.root_dir, "01preprocess_df.csv")
                df.to_csv(out, index=False)
                logging.info(f"Saved preprocess dataframe to {out}")
            print("Preprocess stage completed")
            return df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def time_vars(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            if df is None:
                df = pd.read_csv(self.config.data_path)
            df.loc[df["DAY_OF_WEEK"] > 4, "HOLIDAYS_DAYS_REVENUE"] = (
                df["SALES"] * df["ITEM_PRICE"]
            )
            df.loc[df["DAY_OF_WEEK"] < 5, "HOLIDAYS_DAYS_REVENUE"] = 0

            df.loc[df["DAY_OF_WEEK"] < 4, "WORK_DAYS_SALES"] = 0
            df.loc[df["DAY_OF_WEEK"] > 5, "WORK_DAYS_SALES"] = df["SALES"]

            df.loc[df["DAY_OF_WEEK"] > 4, "HOLIDAYS_DAYS_SALES"] = df["SALES"]
            df.loc[df["DAY_OF_WEEK"] < 5, "HOLIDAYS_DAYS_SALES"] = 0

            out = os.path.join(self.config.root_dir, "02time_vars.csv")
            if self.config.root_dir:
                df.to_csv(out, index=False)
                logging.info(f"Saved time vars dataframe to {out}")
            print("Time vars stage completed")
            return df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def cash_vars(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            if df is None:
                df = pd.read_csv(self.config.data_path)
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df["REVENUE"] = df["ITEM_PRICE"] * df["SALES"]
            df["UNIQUE_DAYS_WITH_SALES"] = df["DATE"]
            df["TOTAL_TRANSACTIONS"] = df["SALES"]
            df["MONTH_DAY"] = df["DATE"].dt.month
            out = os.path.join(self.config.root_dir, "03cash_vars.csv")
            if self.config.root_dir:
                df.to_csv(out, index=False)
                logging.info(f"Saved cash vars dataframe to {out}")
            print("Cash vars stage completed")
            return df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def groupby_month(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            if df is None:
                df = pd.read_csv(self.config.data_path)
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df_monthly_agg = (
                df.set_index("DATE")
                .groupby(["UNIQUE_ID"])
                .resample("MS")
                .agg(
                    {
                        "SALES": np.sum,
                        "REVENUE": np.sum,
                        "UNIQUE_DAYS_WITH_SALES": lambda dates: len(set(dates)),
                        "TOTAL_TRANSACTIONS": len,
                        "ITEM_PRICE": np.mean,
                        "HOLIDAYS_DAYS_REVENUE": np.sum,
                        "HOLIDAYS_DAYS_SALES": np.sum,
                        "WORK_DAYS_SALES": np.sum,
                    }
                )
                .rename(
                    columns={
                        "SALES": "MONTHLY_SALES",
                        "REVENUE": "MONTHLY_REVENUE",
                        "ITEM_PRICE": "MONTHLY_MEAN_PRICE",
                        "HOLIDAYS_DAYS_REVENUE": "MONTHLY_HOLIDAYS_DAYS_REVENUE",
                        "HOLIDAYS_DAYS_SALES": "MONTHLY_HOLIDAYS_DAYS_SALES",
                        "WORK_DAYS_SALES": "MONTHLY_WORK_DAYS_SALES",
                    }
                )
                .reset_index()
            )
            out = os.path.join(self.config.root_dir, "04df_monthly_agg.csv")
            if self.config.root_dir:
                df_monthly_agg.to_csv(out, index=False)
                logging.info(f"Saved monthly agg dataframe to {out}")
            print("Monthly aggregation stage completed")
            return df_monthly_agg
        except Exception as e:
            raise SalesForecastException(e, sys)

    def build_full_range(
        self,
        df: pd.DataFrame | None = None,
        df_monthly_agg: pd.DataFrame | None = None,
        date: str = "2015-10-31",
    ) -> pd.DataFrame:
        try:
            if df is None:
                if not self.config.data_path_df:
                    raise ValueError("data_path_df is required")
                df = pd.read_csv(self.config.data_path_df, index_col=0)
            if df_monthly_agg is None:
                if not self.config.data_path_df_monthly_agg:
                    raise ValueError("data_path_df_monthly_agg is required")
                df_monthly_agg = pd.read_csv(self.config.data_path_df_monthly_agg, index_col=0)
                df_monthly_agg = df_monthly_agg.reset_index()

            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df_monthly_agg["DATE"] = pd.to_datetime(df_monthly_agg["DATE"], errors="coerce")

            df["UNIQUE_ID"] = df["UNIQUE_ID"].astype(str)
            df_monthly_agg["UNIQUE_ID"] = df_monthly_agg["UNIQUE_ID"].astype(str)

            min_date = df["DATE"].min()
            date_prediction = np.datetime64(date)

            unique_id = sorted(df_monthly_agg["UNIQUE_ID"].unique())
            date_range = pd.date_range(min_date, date_prediction, freq="ME")

            cartesian_product = pd.MultiIndex.from_product(
                [date_range, unique_id], names=["DATE", "UNIQUE_ID"]
            )
            full_df = pd.DataFrame(index=cartesian_product).reset_index()
            full_df = pd.merge(df_monthly_agg, full_df, on=["DATE", "UNIQUE_ID"], how="left")

            add_info = df[
                [
                    "UNIQUE_ID",
                    "CITY_NAME",
                    "CITY_ID",
                    "SHOP_NAME",
                    "SHOP_ID",
                    "ITEM_CATEGORY_NAME",
                    "ITEM_CATEGORY_ID",
                    "ITEM_NAME",
                    "ITEM_ID",
                ]
            ].drop_duplicates()

            full_df = pd.merge(full_df, add_info, how="left", on="UNIQUE_ID")

            out = os.path.join(self.config.root_dir, "05full_df.csv")
            if self.config.root_dir:
                full_df.to_csv(out, index=False)
                logging.info(f"Saved full range dataframe to {out}")
            print("Full range stage completed")
            return full_df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def drop_nulls(self, full_df: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            if full_df is None:
                full_df = pd.read_csv(self.config.data_path)
            full_df["MONTHLY_SALES"].fillna(0, inplace=True)
            full_df["MONTHLY_REVENUE"].fillna(0, inplace=True)
            full_df["UNIQUE_DAYS_WITH_SALES"].fillna(0, inplace=True)
            full_df["TOTAL_TRANSACTIONS"].fillna(0, inplace=True)
            full_df["MONTHLY_MEAN_PRICE"].fillna(0, inplace=True)
            full_df["MONTHLY_HOLIDAYS_DAYS_REVENUE"].fillna(0, inplace=True)
            full_df["MONTHLY_HOLIDAYS_DAYS_SALES"].fillna(0, inplace=True)
            full_df["MONTHLY_WORK_DAYS_SALES"].fillna(0, inplace=True)

            out = os.path.join(self.config.root_dir, "06full_df.csv")
            if self.config.root_dir:
                full_df.to_csv(out, index=False)
                logging.info(f"Saved null-filled dataframe to {out}")
            print("Drop nulls stage completed")
            return full_df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def build_ts_vars(
        self,
        full_df: pd.DataFrame,
        gb_list: List[str],
        target_column: str,
        agg_func: Callable,
        agg_func_name: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        assert gb_list[0] == "DATE", "First element must be DATE"
        new_name = "_".join(gb_list + [target_column] + [agg_func_name])
        if verbose:
            print(new_name)
        gb_df_ = (
            full_df.groupby(gb_list)[target_column]
            .apply(agg_func)
            .to_frame()
            .reset_index()
            .rename(columns={target_column: new_name})
        )
        gb_df_[f"{new_name}_LAG1"] = gb_df_.groupby(gb_list[1:])[new_name].transform(
            lambda series: series.shift(1)
        )
        return gb_df_

    def vars_comb(
        self,
        full_df: pd.DataFrame,
        gl: List[str],
        target_column: str,
        agg_func: Callable,
        agg_func_name: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if verbose:
            print(f"Applying transformation: {gl}, {target_column}, {agg_func_name}")
        var = self.build_ts_vars(
            full_df=full_df,
            gb_list=gl,
            target_column=target_column,
            agg_func=agg_func,
            agg_func_name=agg_func_name,
            verbose=verbose,
        )
        return pd.merge(full_df, var, on=gl, how="left")

    def apply_transformations(
        self,
        full_df: pd.DataFrame,
        transformations: List[Tuple[List[str], str, Callable, str]],
        verbose: bool = True,
    ) -> pd.DataFrame:
        for transformation in transformations:
            gl, target_column, agg_func, agg_func_name = transformation
            if verbose:
                print(f"Starting transformation: {gl}, {target_column}, {agg_func_name}")
            full_df = self.vars_comb(
                full_df, gl, target_column, agg_func, agg_func_name, verbose
            )
            if verbose:
                print(f"Completed transformation: {gl}, {target_column}, {agg_func_name}")
        return full_df

    def execute_transformations(self, full_df: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            if full_df is None:
                full_df = pd.read_csv(self.config.data_path)
            transformations = [
                (["DATE", "ITEM_ID"], "MONTHLY_SALES", np.sum, "SUM"),
                (["DATE", "ITEM_ID"], "MONTHLY_HOLIDAYS_DAYS_SALES", np.sum, "SUM"),
                (["DATE", "ITEM_ID"], "TOTAL_TRANSACTIONS", np.sum, "SUM"),
                (["DATE", "ITEM_CATEGORY_ID"], "MONTHLY_HOLIDAYS_DAYS_SALES", np.sum, "SUM"),
            ]
            full_df = self.apply_transformations(full_df, transformations, verbose=True)
            out = os.path.join(self.config.root_dir, "07full_df.csv")
            if self.config.root_dir:
                full_df.to_csv(out, index=False)
                logging.info(f"Saved transformed dataframe to {out}")
            print("Transformations stage completed")
            return full_df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def columns_drop(self, full_df: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            if full_df is None:
                full_df = pd.read_csv(self.config.data_path)
            columns_to_drop = [
                "DATE_ITEM_ID_MONTHLY_SALES_SUM",
                "DATE_ITEM_ID_MONTHLY_HOLIDAYS_DAYS_SALES_SUM",
                "DATE_ITEM_ID_TOTAL_TRANSACTIONS_SUM",
                "DATE_ITEM_CATEGORY_ID_MONTHLY_HOLIDAYS_DAYS_SALES_SUM",
                "MONTHLY_REVENUE",
                "UNIQUE_DAYS_WITH_SALES",
                "TOTAL_TRANSACTIONS",
                "MONTHLY_MEAN_PRICE",
                "CITY_NAME",
                "SHOP_NAME",
                "ITEM_CATEGORY_NAME",
                "ITEM_NAME",
                "MONTHLY_HOLIDAYS_DAYS_SALES",
                "MONTHLY_WORK_DAYS_SALES",
                "SHOP_ID",
            ]
            full_df.drop(columns_to_drop, inplace=True, axis=1)
            full_df = full_df.drop_duplicates()
            out = os.path.join(self.config.root_dir, "08full_df.csv")
            if self.config.root_dir:
                full_df.to_csv(out, index=False)
                logging.info(f"Saved final dataframe to {out}")
            print("Columns drop stage completed")
            return full_df
        except Exception as e:
            raise SalesForecastException(e, sys)


if __name__ == "__main__":
    # Example standalone run
    preprocess_dir = os.path.join("artifacts", "preprocessed")
    os.makedirs(preprocess_dir, exist_ok=True)

    # 01 preprocess
    config = DataTransformationConfig(
        root_dir=preprocess_dir,
        data_path="data/sales_train_merged_.csv",
    )
    transformer = DataTransformation(config)
    df = transformer.preprocess_df(save=False)

    # 02 time vars
    df = transformer.time_vars(df)

    # 03 cash vars
    df = transformer.cash_vars(df)

    # 04 monthly aggregation
    df_monthly = transformer.groupby_month(df)

    # 05 full range build
    full_df = transformer.build_full_range(df=df, df_monthly_agg=df_monthly)

    # 06 drop nulls
    full_df = transformer.drop_nulls(full_df)

    # 07 transformations
    full_df = transformer.execute_transformations(full_df)

    # 08 columns drop
    final_df = transformer.columns_drop(full_df)

    final_path = os.path.join(preprocess_dir, "preprocessed.csv")
    final_df.to_csv(final_path, index=False)
    logging.info(f"Saved final preprocessed dataframe to {final_path}")
    print("All preprocessing stages completed. Output in artifacts/preprocessed/preprocessed.csv")
