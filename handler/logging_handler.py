import mlflow
from helpers import env_helper
from interfaces import EvaluationData
from interfaces.evaluation_data import EvaluationData
class LoggingHandler:
    def __init__(self):
        mlflow.set_tracking_uri(env_helper.MLFLOW_URL)
        mlflow.set_experiment("Product RAG Experiments logging")
        pass

    def log(self, evaluation_data: EvaluationData):
        """Log the product information, ground truths, and prediction to mlflow"""
        product_information = evaluation_data.product_information
        ground_truths = evaluation_data.ground_truths
        prediction = evaluation_data.prediction_result

        try:
            with mlflow.start_run():
                mlflow.log_param("product_name", product_information.name)
                mlflow.log_param("product_description", product_information.description)
                mlflow.log_param("price", product_information.price)
                mlflow.log_param("ground_truths", ground_truths)
                mlflow.log_param("prediction", prediction)
                mlflow.log_param("product_search", evaluation_data.product_search)

                mlflow.log_metric("rouge", evaluation_data.rogue)
                mlflow.log_metric("bert", evaluation_data.bert)
                mlflow.log_metric("bleu", evaluation_data.bleu)
        except Exception as e:
            print(f"Error: {e}")