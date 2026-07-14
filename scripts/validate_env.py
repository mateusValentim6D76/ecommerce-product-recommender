"""Valida o ambiente: versão de Python e dependências principais.

Uso:  uv run python scripts/validate_env.py
"""

import importlib
import sys

REQUIRED_PACKAGES = ["pandas", "numpy", "sklearn", "torch", "mlflow", "fastapi"]
MIN_PYTHON = (3, 13)


def main() -> int:
    ok = True

    if sys.version_info < MIN_PYTHON:
        print(f"[FALHA] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ requerido")
        ok = False
    else:
        print(f"[OK] Python {sys.version.split()[0]}")

    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FALHA] {package} não instalado (rode: uv sync)")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
