"""Run full SalesForecast pipeline end-to-end."""
import sys

from src.exception.exception import SalesForecastException
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_validation import (
    DataValidation,
    DataIngestionArtifact as DVIngestionArtifact,
    DataValidationConfig,
)
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.train_test_split import DataSplit, DataSplitConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation, ModelEvaluationConfig


def run_pipeline():
    try:
        # 1) Data ingestion (from MongoDB)
        ingestion_config = DataIngestionConfig(
            database_name="SalesForecast2026",
            collection_name="forecast",
            feature_store_file_path="artifacts/feature_store/data.csv",
            training_file_path="artifacts/train/train.csv",
            testing_file_path="artifacts/test/test.csv",
            train_test_split_ratio=0.2,
        )
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        # 2) Data validation
        validation_config = DataValidationConfig(
            valid_train_file_path="artifacts/valid/train.csv",
            valid_test_file_path="artifacts/valid/test.csv",
            drift_report_file_path="artifacts/drift/report.yaml",
        )
        validation = DataValidation(
            DVIngestionArtifact(
                trained_file_path=ingestion_artifact.trained_file_path,
                test_file_path=ingestion_artifact.test_file_path,
            ),
            validation_config,
        )
        validation_artifact = validation.initiate_data_validation()
        logging.info("Data validation completed")

        # 3) Data transformation (preprocess all steps)
        transform_config = DataTransformationConfig(
            root_dir="artifacts/preprocessed",
            data_path="data/sales_train_merged_.csv",
        )
        transformer = DataTransformation(transform_config)
        df = transformer.preprocess_df(save=False)
        df = transformer.time_vars(df)
        df = transformer.cash_vars(df)
        df_monthly = transformer.groupby_month(df)
        full_df = transformer.build_full_range(df=df, df_monthly_agg=df_monthly)
        full_df = transformer.drop_nulls(full_df)
        full_df = transformer.execute_transformations(full_df)
        final_df = transformer.columns_drop(full_df)
        final_df.to_csv("artifacts/preprocessed/preprocessed.csv", index=False)
        logging.info("Data transformation completed")

        # 4) Time-series split
        split_config = DataSplitConfig(
            data_path="artifacts/preprocessed/preprocessed.csv",
            root_dir="artifacts/split",
            target_column="MONTHLY_SALES",
            date_column="DATE",
        )
        splitter = DataSplit(split_config)
        splitter.split()
        logging.info("Train/validation/test split completed")

        # 5) Model training
        trainer_config = ModelTrainerConfig(
            root_dir="artifacts/model",
            model_name="xgb_model.joblib",
            data_path_X_train="artifacts/split/X_train.csv",
            data_path_X_valida="artifacts/split/X_valida.csv",
            data_path_Y_train="artifacts/split/Y_train.csv",
            data_path_Y_valida="artifacts/split/Y_valida.csv",
        )
        trainer = ModelTrainer(trainer_config)
        trainer.train_model()
        logging.info("Model training completed")

        # 6) Model evaluation (MLflow + Prometheus)
        eval_config = ModelEvaluationConfig(
            model_path="artifacts/model/xgb_model.joblib",
            x_test_path="artifacts/split/X_test.csv",
            y_test_path="artifacts/split/Y_test.csv",
            report_path="artifacts/evaluation/report.json",
            mlflow_tracking_uri=None,
            mlflow_experiment="SalesForecast",
        )
        evaluator = ModelEvaluation(eval_config)
        evaluator.evaluate()
        logging.info("Model evaluation completed")

        print("Pipeline completed successfully")
    except Exception as e:
        raise SalesForecastException(e, sys)


if __name__ == "__main__":
    run_pipeline()
