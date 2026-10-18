"""
train.py - Обучение моделей.

Запуск отдельно:  python -m src.train   (из корня проекта)
"""

import joblib
import pandas as pd

import config
from src.models import MODEL_FACTORIES, build_pipeline


def load_processed_data() -> tuple:
    """Загружает train/test, сохранённые на шаге генерации данных."""
    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Обучает один пайплайн и сохраняет его на диск."""
    print(f"\n🚀 Обучаем {name}...")
    model = build_pipeline(name)
    # На вход — СЫРЫЕ признаки: feature engineering теперь часть пайплайна,
    # поэтому пороги и кодировки учатся только на X_train.
    model.fit(X_train, y_train)
    joblib.dump(model, config.MODELS_DIR / f"{name}_model.pkl")
    print(f"💾 {name} сохранён")
    return model


def main(model_names: list[str] | None = None) -> dict:
    print("\n" + "=" * 60)
    print("🏋️  ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    X_train, _, y_train, _ = load_processed_data()
    names = model_names or list(MODEL_FACTORIES)
    models = {name: train_model(name, X_train, y_train) for name in names}

    print("\n✅ Все модели обучены и сохранены")
    return models


if __name__ == "__main__":
    main()
