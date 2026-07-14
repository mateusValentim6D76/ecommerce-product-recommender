import os
import random


def seed_everything(seed: int = 42) -> None:
    """Fixa as sementes de aleatoriedade para reprodutibilidade.

    Reprodutibilidade é requisito de MLOps: o mesmo seed + os mesmos
    dados devem gerar o mesmo modelo. torch/numpy são importados de
    forma preguiçosa para não pesar quem só usa o domínio.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
