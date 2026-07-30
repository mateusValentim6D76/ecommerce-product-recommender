"""CLI de recomendação: carrega um modelo e recomenda para um usuário.

Uso:
    uv run python -m recommender.interfaces.cli.recommend \
        --model-name baseline-v1 --user-id 1 --k 10 --data-dir data/raw
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gera recomendações para um usuário")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args(argv)

    from recommender.application.use_cases.generate_recommendation import (
        GenerateRecommendation,
    )
    from recommender.domain.value_objects.user_id import UserId
    from recommender.infrastructure.datasets.movielens_reader import MovieLensReader
    from recommender.infrastructure.datasets.pandas_interaction_repository import (
        PandasInteractionRepository,
    )
    from recommender.infrastructure.persistence.file_pickle_model_repository import (
        FilePickleModelRepository,
    )

    reader = MovieLensReader(
        ratings_path=f"{args.data_dir}/ratings.csv",
        movies_path=f"{args.data_dir}/movies.csv",
    )
    interaction_repository = PandasInteractionRepository.from_reader(reader)
    model_repository = FilePickleModelRepository(args.model_dir)

    use_case = GenerateRecommendation(model_repository, interaction_repository)
    recommendation = use_case.execute(
        model_name=args.model_name, user_id=UserId(args.user_id), k=args.k
    )

    print(f"Recomendações para o usuário {args.user_id}:")
    for item in recommendation.items:
        print(f"  item {item.item_id.value}  (score={item.score.value:.4f})")


if __name__ == "__main__":
    main()
