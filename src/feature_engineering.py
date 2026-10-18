"""
feature_engineering.py - Создание новых признаков из существующих.

Признаки собраны в sklearn-трансформер ChurnFeatureBuilder (fit/transform),
а не в обычную функцию. Зачем:

    Часть признаков зависит от данных: например, "высокий платёж" — это
    платёж выше 75-го перцентиля. Перцентиль нужно посчитать ОДИН раз на
    train (fit) и затем применять к любым новым данным (transform).
    Если считать его заново на test, порог зависит от тестовых данных —
    это утечка, и метрики на test становятся нечестными.

    Трансформер встраивается в Pipeline, поэтому то же самое автоматически
    работает и в кросс-валидации, и при предсказании на новых клиентах.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import TARGET_COLUMN

# Порядок важен: ordinal-признак отражает "длину" контракта.
CONTRACT_ORDER = {"month-to-month": 0, "one_year": 1, "two_year": 2}


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Добавляет к исходным колонкам производные признаки.

    Параметры:
        high_charge_quantile: перцентиль платежа, выше которого платёж "высокий"
        long_tenure_quantile: перцентиль стажа, выше которого клиент "давний"
    """

    def __init__(self, high_charge_quantile: float = 0.75, long_tenure_quantile: float = 0.75):
        self.high_charge_quantile = high_charge_quantile
        self.long_tenure_quantile = long_tenure_quantile

    def fit(self, X: pd.DataFrame, y=None):
        # Атрибуты с "_" в конце — соглашение sklearn: "выучено в fit".
        self.high_charge_threshold_ = X["monthly_charges"].quantile(self.high_charge_quantile)
        self.long_tenure_threshold_ = X["tenure"].quantile(self.long_tenure_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

        # ===== 1. Финансовые метрики =====
        # clip(lower=1): защита от деления на ноль, если появится tenure = 0
        df["avg_monthly_charge"] = df["total_charges"] / df["tenure"].clip(lower=1)
        # > 1: платёж вырос относительно среднего за всё время, < 1: упал
        df["charge_ratio"] = df["monthly_charges"] / df["avg_monthly_charge"]

        # ===== 2. Возрастные группы (не пересекаются) =====
        # В pandas `&` должен соединять два БУЛЕВЫХ условия, каждое в скобках.
        df["is_young"] = (df["age"] < 30).astype(int)
        df["is_middle_age"] = ((df["age"] >= 30) & (df["age"] <= 60)).astype(int)
        df["is_senior"] = (df["age"] > 60).astype(int)

        # ===== 3. Флаги по порогам, выученным на train =====
        df["high_charges"] = (df["monthly_charges"] > self.high_charge_threshold_).astype(int)
        df["long_tenure"] = (df["tenure"] > self.long_tenure_threshold_).astype(int)

        # ===== 4. Взаимодействия признаков =====
        df["internet_high_charge"] = ((df["has_internet"] == 1)
                                      & (df["high_charges"] == 1)).astype(int)
        df["young_monthly"] = ((df["is_young"] == 1)
                               & (df["contract_type"] == "month-to-month")).astype(int)

        # ===== 5. Порядковое кодирование контракта =====
        # У длины контракта есть естественный порядок, поэтому число осмысленно.
        # У способа оплаты порядка нет — он кодируется только через One-Hot.
        df["contract_type_num"] = df["contract_type"].map(CONTRACT_ORDER)

        return df


def get_feature_columns() -> dict:
    """Типы колонок ПОСЛЕ ChurnFeatureBuilder.

    Бинарные флаги (0/1) — числовые: One-Hot для них ничего не добавляет,
    только дублирует колонку.
    """
    categorical = ["payment_method", "contract_type"]
    numerical = [
        "tenure", "monthly_charges", "total_charges", "num_services",
        "has_internet", "has_phone", "age",
        "avg_monthly_charge", "charge_ratio", "contract_type_num",
        "is_young", "is_middle_age", "is_senior",
        "high_charges", "long_tenure", "internet_high_charge", "young_monthly",
    ]
    return {"categorical": categorical, "numerical": numerical}
