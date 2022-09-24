from mininlp.data import Vocab
from mininlp.mlops import MLFlowTrackingCallback
from mininlp.models import QuickClassifierProvider
from mininlp.train import TrainingStats
import mlflow
import numpy as np
from time import time
import random


def _compare_metrics(mlflow_metrics, expected_metrics):
    assert len(mlflow_metrics) == len(expected_metrics)
    assert all(mlflow_m.value == exp_m for mlflow_m, exp_m in zip(mlflow_metrics, expected_metrics))


def test_mlflow_tracking_callback():
    # Use random name because `mlflow.delete_experiment` doesn't completely delete the experiments
    experiment_name = 'exp_test_mlflow_tracking_cb' + str(random.random())
    experiment_no_model_provider_name = 'exp_test_mlflow_tracking_cb_no_model_provider' + str(random.random())
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment_no_model_provider_id = mlflow.create_experiment(experiment_no_model_provider_name)
    
    try:
        mlflow_client = mlflow.tracking.MlflowClient()

        expected_hyperparams = {'hp1': 5, 'hp2': -3.4}
        model_provider = QuickClassifierProvider(Vocab({}, {}), 1, 2)
        cb_model_provider = MLFlowTrackingCallback(experiment_name, expected_hyperparams, model_provider)
        cb_no_model_provider = MLFlowTrackingCallback(experiment_no_model_provider_name, expected_hyperparams) 

        stats_run1 = TrainingStats(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.5]),
            np.array([0.4, 0.6]),
            2,
            8
        )
        stats_run2 = TrainingStats(
            np.array([1.1, 1.2]),
            np.array([1.3, 1.5]),
            np.array([1.4, 1.6]),
            3,
            6
        )        

        for cb in (cb_model_provider, cb_no_model_provider):
            cb.on_train_begin()
            cb.on_train_end(stats_run1)
            cb.on_train_begin()
            cb.on_train_end(stats_run2)

        for exp_name in (experiment_name, experiment_no_model_provider_name):
            runs = mlflow.search_runs(experiment_names=[exp_name], order_by=["attribute.start_time ASC"], output_format="list")
            
            for run, expected_run_stats in zip([runs[0], runs[1]], [stats_run1, stats_run2]):
                actual_tr_loss_history = mlflow_client.get_metric_history(run.info.run_id, 'train_loss')
                _compare_metrics(actual_tr_loss_history, expected_run_stats.train_loss_history)
                actual_tr_metric_history = mlflow_client.get_metric_history(run.info.run_id, 'train_metric')
                _compare_metrics(actual_tr_metric_history, expected_run_stats.train_metric_history)
                actual_val_metric_history = mlflow_client.get_metric_history(run.info.run_id, 'valid_metric')
                _compare_metrics(actual_val_metric_history, expected_run_stats.valid_metric_history)
                assert run.data.params['n_steps'] == str(expected_run_stats.n_steps)
                assert run.data.params['n_epochs'] == str(expected_run_stats.n_epochs)
                # Check if the params that are not n_steps nor n_epochs match
                assert len(run.data.params) == len(expected_hyperparams) + 2
                assert all(str(expected_hyperparams[k]) == run.data.params[k] for k in expected_hyperparams)

                # When model provider is not passed to the callback, embedding_source is not logged
                if exp_name != experiment_no_model_provider_name:
                    assert runs[0].data.tags['embedding_source'] == str(model_provider.embedding_source)

    finally:
        mlflow.delete_experiment(experiment_id)
        mlflow.delete_experiment(experiment_no_model_provider_id)
