"""
feature_engineering.py - Создание новых признаков из существующих.

Признаки оформлены как sklearn-трансформер (fit / transform), а не как функция.
Почему это важно:
  Часть признаков зависит от статистик данных, например порог "дорогой тариф" =
  75-й перцентиль платежа. Первая версия считала этот перцентиль заново на каждом
  наборе: на train по train, на test по test. Это две ошибки сразу:
    1) утечка: статистика тестовых данных влияет на их же признаки;
    2) нестабильность: в проде новые данные приходят по одному клиенту, и
       "перцентиль от одной строки" бессмыслен.
  Трансформер запоминает пороги в fit() на train и применяет их в transform().
  Внутри sklearn Pipeline это происходит автоматически, в том числе в каждом фолде
  кросс-валидации.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config

# Признаки, которые создаёт трансформер, по типам (нужны для предобработки в models.py)
NEW_NUMERICAL = ["avg_monthly_charge", "charge_ratio"]
NEW_BINARY = [
    "is_young",
    "is_middle_age",
    "is_senior",
    "high_charges",
    "long_tenure",
    "internet_high_charge",
    "young_monthly",
]

NUMERICAL_FEATURES = config.NUMERICAL_COLUMNS + NEW_NUMERICAL
BINARY_FEATURES = config.BINARY_COLUMNS + NEW_BINARY
CATEGORICAL_FEATURES = config.CATEGORICAL_COLUMNS


class ChurnFeatures(BaseEstimator, TransformerMixin):
    """Добавляет финансовые метрики, демографические группы и взаимодействия признаков.

    Параметры:
        high_quantile: перцентиль, выше которого платёж/стаж считается "высоким".
    """

    def __init__(self, high_quantile: float = 0.75):
        # По соглашению sklearn в __init__ только сохраняем параметры, без вычислений:
        # на этом построены clone(), GridSearchCV и get_params().
        self.high_quantile = high_quantile

    def fit(self, X: pd.DataFrame, y=None):
        # Атрибуты, выученные из данных, по соглашению sklearn оканчиваются на "_"
        self.high_charge_threshold_ = X["monthly_charges"].quantile(self.high_quantile)
        self.long_tenure_threshold_ = X["tenure"].quantile(self.high_quantile)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()  # не меняем входные данные

        # ========== 1. ФИНАНСОВЫЕ МЕТРИКИ ==========
        # Средний платёж за всё время (clip защищает от деления на 0 при tenure=0)
        df["avg_monthly_charge"] = df["total_charges"] / df["tenure"].clip(lower=1)
        # Текущий платёж / средний: > 1 означает, что тариф подорожал
        df["charge_ratio"] = df["monthly_charges"] / df["avg_monthly_charge"]

        # ========== 2. ДЕМОГРАФИЧЕСКИЕ ГРУППЫ (не пересекаются) ==========
        df["is_young"] = (df["age"] < 30).astype(int)
        df["is_middle_age"] = ((df["age"] >= 30) & (df["age"] <= 60)).astype(int)
        df["is_senior"] = (df["age"] > 60).astype(int)

        # ========== 3. ФЛАГИ ПО ПОРОГАМ, ВЫУЧЕННЫМ НА TRAIN ==========
        df["high_charges"] = (df["monthly_charges"] > self.high_charge_threshold_).astype(int)
        df["long_tenure"] = (df["tenure"] > self.long_tenure_threshold_).astype(int)

        # ========== 4. ВЗАИМОДЕЙСТВИЯ ==========
        df["internet_high_charge"] = (df["has_internet"] & df["high_charges"]).astype(int)
        df["young_monthly"] = (df["is_young"] & (df["contract_type"] == "month-to-month")).astype(
            int
        )

        # Убраны contract_type_num и payment_method_num из первой версии.
        # payment_method — номинальный признак: кодировка банк=0, карта=1, PayPal=2,
        # чек=3 придумывает несуществующий порядок ("чек больше банка"), а после
        # StandardScaler линейная модель воспринимает его как число. Оба признака к
        # тому же дублировали one-hot-кодирование тех же колонок.
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Нужно, чтобы Pipeline и ColumnTransformer знали имена колонок на выходе."""
        return np.asarray(list(self.feature_names_in_) + NEW_NUMERICAL + NEW_BINARY, dtype=object)
