"""CLI de pré-processamento: lê o dataset e mostra o tamanho do split.

Útil para validar que os dados estão no lugar e que o split funciona,
antes de treinar.

Uso:
    uv run python -m recommender.interfaces.cli.preprocess --data-dir data/raw
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pré-processa o dataset MovieLens")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args(argv)

    from recommender.application.use_cases.pre_process_dataset import PreProcessDataset
    from recommender.infrastructure.datasets.movielens_reader import MovieLensReader

    reader = MovieLensReader(
        ratings_path=f"{args.data_dir}/ratings.csv",
        movies_path=f"{args.data_dir}/movies.csv",
    )
    split = PreProcessDataset(reader).execute(test_ratio=args.test_ratio)

    print(f"Treino: {len(split.train)} interações")
    print(f"Teste:  {len(split.test)} interações")


if __name__ == "__main__":
    main()
