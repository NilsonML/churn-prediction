"""
models.py - Сборка пайплайнов "признаки -> предобработка -> модель".

Одна функция build_pipeline() вместо трёх почти одинаковых train_catboost /
train_lightgbm / train_xgboost: добавить новую модель = одна строка в
MODEL_FACTORIES, а не копия функции на 30 строк.

Библиотеки бустинга импортируются ЛЕНИВО (внутри фабрик). Поэтому тесты
и код, которым нужна только одна модель, не требуют установки всех трёх.
"""

from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
from src.feature_engineering import ChurnFeatureBuilder, get_feature_columns


def _catboost():
    import catboost as cb
    return cb.CatBoostClassifier(**config.CATBOOST_PARAMS)


def _lightgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)


def _xgboost():
    import xgboost as xgb
    return xgb.XGBClassifier(**config.XGBOOST_PARAMS)


MODEL_FACTORIES: dict[str, Callable] = {
    "catboost": _catboost,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
}


def create_preprocessor() -> ColumnTransformer:
    """Категориальные -> One-Hot, числовые -> без изменений.

    Почему нет StandardScaler: деревья решений сравнивают признак с порогом,
    и монотонное преобразование (масштабирование) не меняет результат.
    Скейлер нужен линейным моделям и нейросетям, но не бустингу.
    """
    cols = get_feature_columns()
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cols["categorical"]),
            ("num", "passthrough", cols["numerical"]),
        ],
        verbose_feature_names_out=False,  # "contract_type_two_year" вместо "cat__contract_type_two_year"
    )


def build_pipeline(model_name: str) -> Pipeline:
    """Пайплайн целиком: всё, что "учится" на данных, учится только в fit()."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Неизвестная модель {model_name!r}. Доступны: {list(MODEL_FACTORIES)}")
    return Pipeline([
        ("features", ChurnFeatureBuilder()),
        ("preprocessor", create_preprocessor()),
        ("classifier", MODEL_FACTORIES[model_name]()),
    ])
