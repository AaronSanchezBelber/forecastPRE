"""Data validation component."""
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from scipy.stats import ks_2samp
import yaml

from src.exception.exception import SalesForecastException
from src.logging.logger import logging


SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


def read_yaml_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise SalesForecastException(e, sys)


def write_yaml_file(file_path: str, content: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)
    except Exception as e:
        raise SalesForecastException(e, sys)


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationConfig:
    valid_train_file_path: str
    valid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str | None
    invalid_test_file_path: str | None
    drift_report_file_path: str


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SalesForecastException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise SalesForecastException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []

            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)

            logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            return numerical_column_present
        except Exception as e:
            raise SalesForecastException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SalesForecastException(e, sys)

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        try:
            status = True
            report: Dict[str, Any] = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)

                if is_same_dist.pvalue >= threshold:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found,
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise SalesForecastException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                logging.warning("Train dataframe does not contain all columns.")

            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                logging.warning("Test dataframe does not contain all columns.")

            status = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True,
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True,
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            print("Data validation completed")
            return data_validation_artifact
        except Exception as e:
            raise SalesForecastException(e, sys)


if __name__ == "__main__":
    # Example standalone run
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path="artifacts/train/train.csv",
        test_file_path="artifacts/test/test.csv",
    )
    validation_config = DataValidationConfig(
        valid_train_file_path="artifacts/valid/train.csv",
        valid_test_file_path="artifacts/valid/test.csv",
        drift_report_file_path="artifacts/drift/report.yaml",
    )

    validator = DataValidation(ingestion_artifact, validation_config)
    artifact = validator.initiate_data_validation()
    print(artifact)
