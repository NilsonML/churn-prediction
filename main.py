"""
main.py - Точка входа: генерация данных -> кросс-валидация и обучение -> оценка на test.

Использование:
    python main.py                       # весь пайплайн
    python main.py --generate            # только сгенерировать и разбить данные
    python main.py --train --evaluate    # обучить и оценить
    python main.py --models logreg hist_gb --n-samples 20000 --folds 3   # быстрый прогон
"""

import argparse

import config
from src.data_processing import generate_synthetic_data, save_data, split_and_save_data
from src.evaluate import evaluate_all
from src.models import MODEL_BUILDERS
from src.train import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn prediction pipeline")
    parser.add_argument("--generate", action="store_true", help="сгенерировать и разбить данные")
    parser.add_argument("--train", action="store_true", help="кросс-валидация и обучение")
    parser.add_argument("--evaluate", action="store_true", help="оценка на test")
    parser.add_argument("--all", action="store_true", help="все шаги (по умолчанию)")
    parser.add_argument("--n-samples", type=int, default=config.N_SAMPLES)
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_BUILDERS), help="какие модели обучать"
    )
    args = parser.parse_args()
    if not any([args.generate, args.train, args.evaluate]):
        args.all = True
    return args


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    if args.generate or args.all:
        print("\n=== 1. Генерация данных ===")
        df = generate_synthetic_data(n_samples=args.n_samples, random_state=config.RANDOM_STATE)
        save_data(df, "churn.csv")
        split_and_save_data(df)

    if args.train or args.all:
        print("\n=== 2. Кросс-валидация и обучение ===")
        train_main(model_names=args.models, n_folds=args.folds)

    if args.evaluate or args.all:
        print("\n=== 3. Оценка на test ===")
        evaluate_all()


if __name__ == "__main__":
    main()
