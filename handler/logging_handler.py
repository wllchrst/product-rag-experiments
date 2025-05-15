import mlflow
from helpers import env_helper
from interfaces import ProductInformation
from typing import List
class LoggingHandler:
    def __init__(self):
        mlflow.set_tracking_uri(env_helper.MLFLOW_URL)
        mlflow.set_experiment("Product RAG Experiments logging")
        pass

    def log(self, product_information: ProductInformation, ground_truths: List[str], prediction: str, rogue: dict, bert: dict):
        """Log the product information, ground truths, and prediction to mlflow"""
        try:
            with mlflow.start_run():
                mlflow.log_param("product_name", product_information.name)
                mlflow.log_param("product_description", product_information.description)
                mlflow.log_param("price", product_information.price)
                mlflow.log_param("ground_truths", ground_truths)
                mlflow.log_param("prediction", prediction)
                mlflow.log_param("rouge", rogue)
                mlflow.log_param("bert", bert)
        except Exception as e:
            print(f"Error: {e}")