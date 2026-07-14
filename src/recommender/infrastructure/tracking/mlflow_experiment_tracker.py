import mlflow

from recommender.domain.entities.experiment import Experiment


class MlflowExperimentTracker:
    """Adapter: implementa ExperimentTracker sobre o MLflow.

    Recebe um Experiment já finalizado e registra hiperparâmetros,
    métricas e metadados num run do MLflow. O núcleo não importa
    mlflow — só esta classe, na borda da infraestrutura.
    """

    def __init__(
        self,
        experiment_name: str = "recommender",
        tracking_uri: str | None = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def track(self, experiment: Experiment) -> None:
        with mlflow.start_run(run_name=experiment.name):
            mlflow.set_tag("model_name", experiment.model_name)
            mlflow.log_params(experiment.hyperparameters.values)
            if experiment.metrics is not None:
                mlflow.log_metrics(experiment.metrics.values)
