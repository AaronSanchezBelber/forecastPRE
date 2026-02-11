# Model training component for SalesForecast
# Imports
import os  # filesystem operations
import sys  # system info for exceptions
from dataclasses import dataclass  # config dataclass
from datetime import datetime as dt  # timestamps for model versioning

import pandas as pd  # data handling
import xgboost as xgb  # model
import joblib  # model persistence
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # metrics
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway  # metrics push

from src.exception.exception import SalesForecastException  # custom exception
from src.logging.logger import logging  # project logger


# Configuration for model trainer
@dataclass
class ModelTrainerConfig:
    root_dir: str  # folder where model will be saved
    model_name: str  # model filename
    data_path_X_train: str  # path to X_train
    data_path_X_valida: str  # path to X_valid
    data_path_Y_train: str  # path to y_train
    data_path_Y_valida: str  # path to y_valid


# Model trainer class
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        try:
            # store config
            self.config = config
        except Exception as e:
            # raise custom exception
            raise SalesForecastException(e, sys)

    def train_model(self):
        try:
            # read train/validation datasets
            X_train = pd.read_csv(self.config.data_path_X_train)
            X_valida = pd.read_csv(self.config.data_path_X_valida)
            Y_train = pd.read_csv(self.config.data_path_Y_train)
            Y_valida = pd.read_csv(self.config.data_path_Y_valida)

            # define model
            model = xgb.XGBRegressor(
                eval_metric="rmse",
                early_stopping_rounds=10,
                seed=175,
            )

            # log model name and training start
            model_name = str(model).split("(")[0]
            logging.info(f"Training model: {model_name}")

            # create a timestamp for traceability
            day = str(dt.now()).split()[0].replace("-", "_")
            hour = str(dt.now()).split()[1].replace(":", "_").split(".")[0]
            _ = str(day) + "_" + str(hour)

            # fit model
            model.fit(
                X_train,
                Y_train,
                eval_set=[(X_train, Y_train), (X_valida, Y_valida)],
                verbose=True,
            )

            # basic validation metrics
            y_val_pred = model.predict(X_valida)
            y_val_true = Y_valida.values.ravel()
            rmse = mean_squared_error(y_val_true, y_val_pred) ** 0.5
            mae = mean_absolute_error(y_val_true, y_val_pred)
            r2 = r2_score(y_val_true, y_val_pred)

            # ensure output directory exists
            os.makedirs(self.config.root_dir, exist_ok=True)

            # save model
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(model, model_path)
            logging.info(f"Model saved at: {model_path}")

            # push metrics to Prometheus Pushgateway if configured
            pushgateway_url = os.getenv("PUSHGATEWAY_URL")
            if pushgateway_url:
                registry = CollectorRegistry()
                Gauge("salesforecast_val_rmse", "Validation RMSE", registry=registry).set(rmse)
                Gauge("salesforecast_val_mae", "Validation MAE", registry=registry).set(mae)
                Gauge("salesforecast_val_r2", "Validation R2", registry=registry).set(r2)
                push_to_gateway(pushgateway_url, job="model_trainer", registry=registry)

            # return trained model
            return model
        except Exception as e:
            # raise custom exception
            raise SalesForecastException(e, sys)


# Optional local run for quick testing
if __name__ == "__main__":
    # example config (update paths as needed)
    config = ModelTrainerConfig(
        root_dir="artifacts/model",
        model_name="xgb_model.joblib",
        data_path_X_train="artifacts/split/X_train.csv",
        data_path_X_valida="artifacts/split/X_valida.csv",
        data_path_Y_train="artifacts/split/Y_train.csv",
        data_path_Y_valida="artifacts/split/Y_valida.csv",
    )

    # train model
    trainer = ModelTrainer(config)
    trainer.train_model()
    print("Model training completed")
