"""Model evaluation with MLflow tracking."""
import os
import sys
import json
from dataclasses import dataclass

from dotenv import load_dotenv

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.exception.exception import SalesForecastException
from src.logging.logger import logging

load_dotenv()


@dataclass
class ModelEvaluationConfig:
    model_path: str
    x_test_path: str
    y_test_path: str
    report_path: str
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "SalesForecast"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        try:
            # Load model and data
            model = joblib.load(self.config.model_path)
            X_test = pd.read_csv(self.config.x_test_path)
            y_test = pd.read_csv(self.config.y_test_path)

            # Align shapes
            if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
                y_true = y_test.iloc[:, 0].values
            else:
                y_true = y_test.values.ravel()

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics = {"rmse": rmse, "mae": mae, "r2": r2}

            # Save report
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
            with open(self.config.report_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            logging.info(f"Model evaluation report saved: {self.config.report_path}")

            # MLflow tracking
            import mlflow as mlflow_lib
            tracking_uri = (
                self.config.mlflow_tracking_uri
                or os.getenv("MLFLOW_TRACKING_URI")
                or "file:./mlruns"
            )
            try:
                mlflow_lib.set_tracking_uri(tracking_uri)
                mlflow_lib.set_experiment(self.config.mlflow_experiment)

                with mlflow_lib.start_run():
                    mlflow_lib.log_metrics(metrics)
                    mlflow_lib.log_param("model_path", self.config.model_path)
                    # Log model if available
                    try:
                        import mlflow.xgboost

                        mlflow_lib.xgboost.log_model(model, "model")
                    except Exception:
                        import mlflow.sklearn

                        mlflow_lib.sklearn.log_model(model, "model")
            except Exception as mlflow_err:
                logging.warning(
                    "MLflow tracking skipped due to error: %s. "
                    "If using a server, ensure it's running or unset MLFLOW_TRACKING_URI.",
                    mlflow_err,
                )

            # push metrics to Prometheus Pushgateway if configured
            pushgateway_url = os.getenv("PUSHGATEWAY_URL")
            if pushgateway_url:
                registry = CollectorRegistry()
                Gauge("salesforecast_eval_rmse", "Eval RMSE", registry=registry).set(rmse)
                Gauge("salesforecast_eval_mae", "Eval MAE", registry=registry).set(mae)
                Gauge("salesforecast_eval_r2", "Eval R2", registry=registry).set(r2)
                push_to_gateway(pushgateway_url, job="model_evaluation", registry=registry)

            print("Model evaluation completed")
            return metrics
        except Exception as e:
            raise SalesForecastException(e, sys)


if __name__ == "__main__":
    config = ModelEvaluationConfig(
        model_path="artifacts/model/xgb_model.joblib",
        x_test_path="artifacts/split/X_test.csv",
        y_test_path="artifacts/split/Y_test.csv",
        report_path="artifacts/evaluation/report.json",
        mlflow_tracking_uri=None,
        mlflow_experiment="SalesForecast",
    )

    evaluator = ModelEvaluation(config)
    evaluator.evaluate()
