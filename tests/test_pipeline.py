"""Тесты пайплайна. Бустинги заменены на модели sklearn (monkeypatch),
чтобы тесты не требовали catboost/lightgbm/xgboost и шли за секунды."""

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from src import models


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(models, "MODEL_FACTORIES", {
        "fake_gbm": lambda: HistGradientBoostingClassifier(max_iter=50, random_state=0),
    })


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        models.build_pipeline("no_such_model")


def test_pipeline_trains_on_raw_features(small_df):
    X, y = small_df.drop(columns="churn"), small_df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    pipe = models.build_pipeline("fake_gbm").fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)
    assert roc_auc_score(y_te, proba) > 0.9                    # таргет задан правилом -> легко


def test_pipeline_works_in_cross_validation(small_df):
    """Главный плюс трансформера: пороги пересчитываются внутри каждого фолда."""
    X, y = small_df.drop(columns="churn"), small_df["churn"]
    scores = cross_val_score(models.build_pipeline("fake_gbm"), X, y, cv=3, scoring="roc_auc")
    assert len(scores) == 3 and scores.min() > 0.9


def test_feature_names_available_for_shap(small_df):
    X, y = small_df.drop(columns="churn"), small_df["churn"]
    pipe = models.build_pipeline("fake_gbm").fit(X, y)
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    assert pipe[:-1].transform(X.head()).shape[1] == len(names)
    assert "contract_type_two_year" in names
