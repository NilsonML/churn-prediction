import pandas as pd

from src.feature_engineering import ChurnFeatureBuilder, get_feature_columns


def _row(**kw):
    base = dict(tenure=10, monthly_charges=50.0, total_charges=500.0, num_services=3,
                has_internet=1, has_phone=1, payment_method="paypal",
                contract_type="one_year", age=40)
    base.update(kw)
    return base


def test_age_groups_are_disjoint_and_cover_all_ages():
    X = pd.DataFrame([_row(age=a) for a in (18, 29, 30, 31, 40, 59, 60, 61, 79)])
    out = ChurnFeatureBuilder().fit(X).transform(X)
    groups = out[["is_young", "is_middle_age", "is_senior"]]
    assert (groups.sum(axis=1) == 1).all()            # ровно одна группа у каждого
    assert out.loc[out["age"].isin([30, 31, 40, 59, 60]), "is_middle_age"].eq(1).all()


def test_thresholds_are_learned_on_train_only():
    """Порог 'высокого платежа' считается в fit и НЕ пересчитывается на новых данных."""
    train = pd.DataFrame([_row(monthly_charges=c) for c in range(20, 120)])
    test = pd.DataFrame([_row(monthly_charges=200.0)] * 10)   # все платежи огромные
    fb = ChurnFeatureBuilder().fit(train)
    threshold = fb.high_charge_threshold_
    out = fb.transform(test)
    assert fb.high_charge_threshold_ == threshold              # transform не меняет состояние
    assert out["high_charges"].eq(1).all()                     # если бы порог считался на test — было бы 0


def test_target_is_dropped_and_columns_match_spec(small_df):
    out = ChurnFeatureBuilder().fit(small_df).transform(small_df)
    assert "churn" not in out.columns
    spec = get_feature_columns()
    assert set(spec["categorical"] + spec["numerical"]) <= set(out.columns)


def test_transform_does_not_modify_input(small_df):
    before = small_df.copy()
    ChurnFeatureBuilder().fit(small_df).transform(small_df)
    pd.testing.assert_frame_equal(small_df, before)
