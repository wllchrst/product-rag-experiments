import mlflow
import numpy as np
from helpers import env_helper
from datetime import datetime
from interfaces import EvaluationData
from interfaces.evaluation_data import EvaluationData
class LoggingHandler:
    def __init__(self):
        mlflow.set_tracking_uri(env_helper.MLFLOW_URL)
        mlflow.set_experiment("Product RAG Experiments logging")
        pass

    def insert_metric_from_dict(self, data: dict, key_prefix: str):
        print(data)
        for key, value in data.items():
            value_to_insert = value

            if isinstance(value_to_insert, list):
                value_to_insert = np.mean(value_to_insert)
            elif isinstance(value_to_insert, str):
                continue
            
            mlflow.log_metric(f"{key_prefix}_{key}", value_to_insert)

    def log(self, evaluation_data: EvaluationData):
        """Log the product information, ground truths, and prediction to mlflow"""
        product_information = evaluation_data.product_information
        ground_truths = evaluation_data.ground_truths
        prediction = evaluation_data.prediction_result

        try:
            experiment_name = datetime.now().strftime("experiment-%Y%m%d-%H%M%S")

            with mlflow.start_run(run_name=experiment_name):
                mlflow.log_param("product_name", product_information.name)
                mlflow.log_param("product_description", product_information.description)
                mlflow.log_param("price", product_information.price)
                mlflow.log_param("ground_truths", ground_truths)
                mlflow.log_param("prediction", prediction)
                mlflow.log_param("product_search", evaluation_data.product_search)
                mlflow.log_param("method", evaluation_data.method)

                self.insert_metric_from_dict(evaluation_data.bert, 'bert')
                self.insert_metric_from_dict(evaluation_data.bleu, 'bleu')
                self.insert_metric_from_dict(evaluation_data.rouge, 'rouge')

        except Exception as e:
            print(f"Error: {e}")