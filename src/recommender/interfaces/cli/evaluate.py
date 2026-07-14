"""CLI de avaliação: carrega um modelo salvo e avalia sobre o teste.

Uso:
    uv run python -m recommender.interfaces.cli.evaluate \
        --model-name baseline-v1 --data-dir data/raw --k 10
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Avalia um modelo salvo")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    from recommender.application.use_cases.evaluate_model import EvaluateModel
    from recommender.application.use_cases.pre_process_dataset import PreProcessDataset
    from recommender.infrastructure.datasets.movielens_reader import MovieLensReader
    from recommender.infrastructure.evaluation.ranking_metrics import (
        RankingMetricCalculator,
    )
    from recommender.infrastructure.persistence.file_pickle_model_repository import (
        FilePickleModelRepository,
    )

    reader = MovieLensReader(
        ratings_path=f"{args.data_dir}/ratings.csv",
        movies_path=f"{args.data_dir}/movies.csv",
    )
    split = PreProcessDataset(reader).execute(test_ratio=args.test_ratio)

    model = FilePickleModelRepository(args.model_dir).get(args.model_name)
    if model is None:
        raise SystemExit(f"Modelo '{args.model_name}' não encontrado em {args.model_dir}")

    metrics = EvaluateModel(RankingMetricCalculator()).execute(
        model=model, test_interactions=split.test, k=args.k
    )
    print(f"Métricas de {args.model_name}: {metrics.values}")


if __name__ == "__main__":
    main()
