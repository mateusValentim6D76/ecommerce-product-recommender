"""CLI de treino: amarra o pipeline inteiro (composição).

Fluxo: ler dados -> split -> treinar -> avaliar -> registrar experimento.
Usa as portas/adapters já prontos; o 'miolo' de ML (trainer e métricas)
é o que VOCÊ implementa nos stubs.

Uso:
    uv run python -m recommender.interfaces.cli.train \
        --model-type baseline --model-name baseline-v1 --data-dir data/raw
"""

import argparse
from datetime import UTC, datetime


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treina um modelo de recomendação")
    parser.add_argument("--model-type", required=True, choices=["baseline", "pytorch"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--mlflow-uri", default=None)
    args = parser.parse_args(argv)

    from recommender.application.use_cases.evaluate_model import EvaluateModel
    from recommender.application.use_cases.pre_process_dataset import PreProcessDataset
    from recommender.application.use_cases.train_model import TrainModel
    from recommender.domain.entities.experiment import Experiment
    from recommender.domain.value_objects.hyperparameters import Hyperparameters
    from recommender.infrastructure.datasets.movielens_reader import MovieLensReader
    from recommender.infrastructure.evaluation.ranking_metrics import (
        RankingMetricCalculator,
    )
    from recommender.infrastructure.models.model_factory import create_trainer
    from recommender.infrastructure.persistence.file_pickle_model_repository import (
        FilePickleModelRepository,
    )
    from recommender.infrastructure.tracking.mlflow_experiment_tracker import (
        MlflowExperimentTracker,
    )
    from recommender.shared.seed import seed_everything

    seed_everything()

    hyperparameters = Hyperparameters(
        {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "embedding_dim": args.embedding_dim,
            "k": args.k,
        }
    )

    reader = MovieLensReader(
        ratings_path=f"{args.data_dir}/ratings.csv",
        movies_path=f"{args.data_dir}/movies.csv",
    )
    split = PreProcessDataset(reader).execute(test_ratio=args.test_ratio)

    trainer = create_trainer(args.model_type)
    model_repository = FilePickleModelRepository(args.model_dir)

    experiment = Experiment(
        name=args.model_name,
        model_name=args.model_type,
        hyperparameters=hyperparameters,
    )
    experiment.start_at(datetime.now(UTC))

    model = TrainModel(trainer, model_repository).execute(
        model_name=args.model_name,
        interactions=split.train,
        hyperparameters=hyperparameters,
    )

    metrics = EvaluateModel(RankingMetricCalculator()).execute(
        model=model, test_interactions=split.test, k=args.k
    )
    experiment.finish(metrics=metrics, finished_at=datetime.now(UTC))

    MlflowExperimentTracker(tracking_uri=args.mlflow_uri).track(experiment)

    print(f"Treino concluído: {args.model_name}")
    print(f"Métricas: {metrics.values}")


if __name__ == "__main__":
    main()
