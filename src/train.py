"""
train.py - Кросс-валидация и обучение моделей.

Схема оценки:
  1. На train: стратифицированная k-fold кросс-валидация каждой модели.
     Даёт среднее И разброс метрики. По ней сравниваем и выбираем модели.
  2. Обучение каждой модели на всём train и сохранение.
  3. Test не трогаем до evaluate.py: финальная проверка выбранного, а не выбор.

Метрики:
  ROC-AUC — насколько хорошо модель ранжирует клиентов по риску (не зависит от порога).
  PR-AUC (average precision) — то же, но с фокусом на редкий класс: при 25% оттока
  "случайная" модель имеет PR-AUC 0.25, а не 0.5, и это честнее показывает пользу.
"""

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

import config
from src.data_processing import load_processed_data
from src.models import build_models

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def cross_validate_models(
    models: dict, X: pd.DataFrame, y: pd.Series, n_folds: int = config.CV_FOLDS
) -> pd.DataFrame:
    """k-fold CV для каждой модели. Возвращает таблицу mean ± std по фолдам."""
    # Одни и те же фолды для всех моделей (фиксированный random_state):
    # иначе разница между моделями смешается с разницей между разбиениями.
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, model in models.items():
        print(f"CV {name} ({n_folds} фолдов)...")
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name, "fit_time_s": scores["fit_time"].mean()}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
        print(f"  ROC-AUC {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def main(model_names: list[str] | None = None, n_folds: int = config.CV_FOLDS) -> None:
    config.ensure_dirs()
    X_train, _, y_train, _ = load_processed_data()
    models = build_models(model_names)

    cv_results = cross_validate_models(models, X_train, y_train, n_folds)
    cv_results.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)
    print(f"\nРезультаты CV сохранены в {config.REPORTS_DIR / 'cv_results.csv'}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, config.MODELS_DIR / f"{name}.pkl")
    print(f"Модели обучены на всём train и сохранены в {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
