"""
models.py - Все модели проекта как sklearn-пайплайны: признаки -> предобработка -> модель.

Почему пайплайн целиком, а не "сначала признаки, потом модель":
  в кросс-валидации трансформеры (пороги признаков, StandardScaler) должны учиться
  только на обучающих фолдах. Pipeline гарантирует это автоматически; при ручной
  предобработке до CV статистики валидационного фолда утекают в обучение.

Бейзлайн
  Логистическая регрессия. Без бейзлайна нельзя сказать, стоят ли бустинги своей
  сложности: если они выигрывают 0.002 AUC, проще и надёжнее линейная модель.

Бустинги
  CatBoost, LightGBM, XGBoost опциональны: если библиотека не установлена, модель
  пропускается с сообщением, а остальные работают. HistGradientBoosting из sklearn
  доступен всегда и служит "бустингом без дополнительных зависимостей".
"""

from collections.abc import Callable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.feature_engineering import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ChurnFeatures,
)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Предобработка после ChurnFeatures.

    scale: стандартизировать ли числовые признаки. Линейной модели это нужно
        (регуляризация штрафует коэффициенты одинаково, только если признаки в одном
        масштабе). Деревьям безразлично: они делят по порогам, а не по расстояниям.
    Бинарные флаги проходят как есть: 0/1 уже готовый признак, раньше их
    one-hot-кодировали как категории, что ничего не давало.
    One-hot без drop="first": при L2-регуляризации мультиколлинеарность полного
    набора дамми-переменных не мешает, а деревьям она безразлична.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _logreg() -> Pipeline:
    return Pipeline(
        [
            ("features", ChurnFeatures()),
            ("preprocessor", make_preprocessor(scale=True)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def _hist_gb() -> Pipeline:
    return Pipeline(
        [
            ("features", ChurnFeatures()),
            ("preprocessor", make_preprocessor(scale=False)),
            ("classifier", HistGradientBoostingClassifier(**config.HIST_GB_PARAMS)),
        ]
    )


def _lightgbm() -> Pipeline:
    import lightgbm as lgb

    return Pipeline(
        [
            ("features", ChurnFeatures()),
            ("preprocessor", make_preprocessor(scale=False)),
            ("classifier", lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)),
        ]
    )


def _xgboost() -> Pipeline:
    import xgboost as xgb

    return Pipeline(
        [
            ("features", ChurnFeatures()),
            ("preprocessor", make_preprocessor(scale=False)),
            ("classifier", xgb.XGBClassifier(**config.XGBOOST_PARAMS)),
        ]
    )


class CatBoostNative(ClassifierMixin, BaseEstimator):
    """Обёртка над CatBoostClassifier с родной обработкой категорий.

    Зачем обёртка: cross_validate клонирует модель через sklearn.clone(), а clone
    проверяет, что конструктор сохраняет параметры без изменений. CatBoost копирует
    список cat_features внутри, и clone падает с "Cannot clone object ... cat_features"
    (нашёл CI). Здесь параметры хранятся как есть, а CatBoost создаётся в fit().
    """

    def __init__(self, cat_features: tuple[str, ...] = (), params: dict | None = None):
        self.cat_features = cat_features
        self.params = params

    def fit(self, X, y):
        import catboost as cb

        self.model_ = cb.CatBoostClassifier(
            cat_features=list(self.cat_features), **(self.params or {})
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def _catboost() -> Pipeline:
    import catboost  # noqa: F401  проверяем наличие библиотеки сразу, а не в fit

    # Без one-hot: главное преимущество CatBoost — собственная обработка категорий
    # (ordered target statistics). Первая версия подавала ему уже закодированные
    # данные, и это преимущество терялось.
    return Pipeline(
        [
            ("features", ChurnFeatures()),
            (
                "classifier",
                CatBoostNative(
                    cat_features=tuple(CATEGORICAL_FEATURES), params=config.CATBOOST_PARAMS
                ),
            ),
        ]
    )


MODEL_BUILDERS: dict[str, Callable[[], Pipeline]] = {
    "logreg": _logreg,
    "hist_gb": _hist_gb,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "catboost": _catboost,
}


def build_models(names: list[str] | None = None) -> dict[str, Pipeline]:
    """Собирает модели по именам. Недоступные библиотеки пропускаются с сообщением."""
    models = {}
    for name in names or list(MODEL_BUILDERS):
        if name not in MODEL_BUILDERS:
            raise ValueError(f"Неизвестная модель {name!r}. Доступны: {list(MODEL_BUILDERS)}")
        try:
            models[name] = MODEL_BUILDERS[name]()
        except ImportError as e:
            print(f"Пропускаю {name}: {e.name} не установлен (pip install {e.name})")
    return models
