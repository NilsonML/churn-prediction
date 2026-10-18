"""Тесты инвариантов, на которых держится честность оценки.

Каждый тест ловит конкретную ошибку, которая уже была в проекте или типична для ML:
утечку ответа в признаки, статистики test в признаках, недетерминированные данные.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
from src.data_processing import generate_synthetic_data, split_features_target
from src.feature_engineering import BINARY_FEATURES, NUMERICAL_FEATURES, ChurnFeatures
from src.models import build_models


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return generate_synthetic_data(n_samples=5_000, random_state=0)


def test_generation_is_deterministic(data):
    pd.testing.assert_frame_equal(data, generate_synthetic_data(n_samples=5_000, random_state=0))


def test_target_is_stochastic_not_a_rule(data):
    """Таргет сэмплирован из вероятности, а не задан порогом: у клиентов с одинаково
    высоким риском исходы должны различаться. Иначе модели снова "выучат формулу"."""
    high_risk = data[data[config.ORACLE_COLUMN] > 0.5]
    assert 0 < high_risk[config.TARGET_COLUMN].mean() < 1
    assert 0.15 < data[config.TARGET_COLUMN].mean() < 0.35


def test_oracle_and_target_are_not_features(data):
    X, _, _ = split_features_target(data)
    assert config.TARGET_COLUMN not in X.columns
    assert config.ORACLE_COLUMN not in X.columns


def test_thresholds_are_learned_on_train_only(data):
    """Порог high_charges выучен на train и не меняется от данных, на которых применяется."""
    X, _, _ = split_features_target(data)
    fe = ChurnFeatures().fit(X)
    threshold = fe.high_charge_threshold_

    # "Test" из одних дорогих клиентов: его собственный перцентиль был бы другим
    expensive = X[X["monthly_charges"] > 140]
    out = fe.transform(expensive)
    assert fe.high_charge_threshold_ == threshold
    assert out["high_charges"].all()


def test_single_row_transform_works(data):
    """В проде клиент приходит по одному: признаки должны считаться и для одной строки."""
    X, _, _ = split_features_target(data)
    fe = ChurnFeatures().fit(X)
    out = fe.transform(X.iloc[[0]])
    assert len(out) == 1
    assert not out[NUMERICAL_FEATURES + BINARY_FEATURES].isna().any().any()


def test_age_groups_are_disjoint_and_complete(data):
    X, _, _ = split_features_target(data)
    out = ChurnFeatures().fit_transform(X)
    groups = out[["is_young", "is_middle_age", "is_senior"]].sum(axis=1)
    assert (groups == 1).all()


def test_baseline_learns_signal_but_not_beyond_oracle(data):
    """Логрег должен заметно превзойти случайность и не может стабильно обогнать оракула."""
    X, y, oracle = split_features_target(data)
    X_tr, X_te, y_tr, y_te, _, o_te = train_test_split(
        X, y, oracle, test_size=0.3, random_state=0, stratify=y
    )
    model = build_models(["logreg"])["logreg"].fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    oracle_auc = roc_auc_score(y_te, o_te)
    assert auc > 0.65
    assert auc < oracle_auc + 0.02  # небольшой запас на шум конечной выборки
    assert np.isfinite(auc)
