from mininlp.models import BaseModelProvider
from mininlp.train import TrainingCallback, TrainingStats
import mlflow
from typing import Type


class MLFlowTrackingCallback(TrainingCallback):
    """Callback that logs the hyperparameters and metrics of a experiment using MLFlow.

    It's associated to a single experiment but multiple runs of the same experiment may be run.
    
    Args:
        experiment_name: name of MLFlow experiment
        hyperparams: dict that contains the hyperparameter names as keys and the values as values
        model_provider: model provider used to create the models trained under this experiment
    """
    def __init__(self, experiment_name:str, hyperparams:dict, model_provider:Type[BaseModelProvider]=None):
        self.run = None
        self.experiment_name = experiment_name
        self.hyperparams = hyperparams
        self.model_provider = model_provider
        
    def on_train_begin(self):
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()
        mlflow.log_params(self.hyperparams)
        if (self.model_provider is not None) and (hasattr(self.model_provider, "embedding_source")):
            mlflow.set_tag("embedding_source", self.model_provider.embedding_source)

    def on_train_end(self, stats:TrainingStats):
        if self.run is not None:
            for i, tr_loss in enumerate(stats.train_loss_history):
                mlflow.log_metric("train_loss", tr_loss, i)
            for i, tr_metric in enumerate(stats.train_metric_history):
                mlflow.log_metric("train_metric", tr_metric, i)
            for i, val_metric in enumerate(stats.valid_metric_history):
                mlflow.log_metric("valid_metric", val_metric, i)
            mlflow.log_param("n_steps", stats.n_steps)
            mlflow.log_param("n_epochs", stats.n_epochs)
            mlflow.end_run()
            self.run = None
