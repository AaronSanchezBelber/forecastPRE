"""Data ingestion component."""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.exception.exception import SalesForecastException
from src.logging.logger import logging

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


@dataclass
class DataIngestionConfig:
    database_name: str
    collection_name: str
    feature_store_file_path: str
    training_file_path: str
    testing_file_path: str
    train_test_split_ratio: float = 0.2


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SalesForecastException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL no esta definido en el entorno.")

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise SalesForecastException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            test_dir = os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Exported train and test file path.")
        except Exception as e:
            raise SalesForecastException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise SalesForecastException(e, sys)


if __name__ == "__main__":
    config = DataIngestionConfig(
        database_name="SalesForecast2026",
        collection_name="forecast",
        feature_store_file_path="artifacts/feature_store/data.csv",
        training_file_path="artifacts/train/train.csv",
        testing_file_path="artifacts/test/test.csv",
        train_test_split_ratio=0.2,
    )

    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion()

    print("Data Ingestion completed")
    print(artifact)
