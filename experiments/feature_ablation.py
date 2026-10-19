"""Абляция: сколько дают сгенерированные признаки линейной модели и бустингу?

Вопрос: что важнее, выбор модели или признаки? Сравниваем 2 модели x 2 набора
признаков (сырые / сырые + ChurnFeatures) на ОДНИХ И ТЕХ ЖЕ 5 фолдах.

Почему считаем парную разницу по фолдам, а не просто сравниваем средние:
разброс между фолдами (~0.004 AUC) в основном общий для всех моделей (лёгкий или
трудный фолд). Разница на одном фолде убирает этот общий шум, и вывод "A лучше B"
становится надёжнее.

Запуск (после `python main.py --generate`):
    python experiments/feature_ablation.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # корень репозитория

import config  # noqa: E402
from src.data_processing import load_processed_data  # noqa: E402
from src.models import build_models  # noqa: E402


def raw_pipeline(classifier) -> Pipeline:
    """Те же модели, но без ChurnFeatures: только исходные колонки."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERICAL_COLUMNS),
            ("bin", "passthrough", config.BINARY_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline([("preprocessor", pre), ("classifier", classifier)])


def main() -> None:
    X, _, y, _ = load_processed_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    engineered = build_models(["logreg", "hist_gb"])
    variants = {
        ("logreg", "raw"): raw_pipeline(LogisticRegression(max_iter=1000)),
        ("logreg", "engineered"): engineered["logreg"],
        ("hist_gb", "raw"): raw_pipeline(HistGradientBoostingClassifier(**config.HIST_GB_PARAMS)),
        ("hist_gb", "engineered"): engineered["hist_gb"],
    }
    fold_scores = {
        key: cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        for key, model in variants.items()
    }

    rows = []
    for model in ("logreg", "hist_gb"):
        raw, eng = fold_scores[(model, "raw")], fold_scores[(model, "engineered")]
        diff = eng - raw
        rows.append(
            {
                "model": model,
                "roc_auc_raw": raw.mean(),
                "roc_auc_engineered": eng.mean(),
                "gain_mean": diff.mean(),
                "gain_std": diff.std(),
                "folds_improved": f"{int((diff > 0).sum())}/{len(diff)}",
            }
        )
    result = pd.DataFrame(rows)
    config.ensure_dirs()
    result.to_csv(config.REPORTS_DIR / "feature_ablation.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(result.to_string(index=False))
    print("\ngain = engineered - raw на каждом фолде (парная разница)")


if __name__ == "__main__":
    main()
