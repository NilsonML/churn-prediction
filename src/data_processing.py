"""
data_processing.py - Генерация синтетических данных и разбиение на train/test.

Почему данные синтетические и как сделать их честными
------------------------------------------------------
Первая версия задавала отток жёстким правилом `сумма факторов > 0.5`. Таргет был
детерминированной функцией признаков, модели восстанавливали формулу и получали
ROC-AUC 0.996. Такая цифра говорит не о качестве модели, а о том, что задача
нереалистично лёгкая; на интервью это первый вопрос.

Теперь отток СЛУЧАЕН: каждый клиент уходит с вероятностью p = sigmoid(logit),
где logit зависит от признаков. Как в жизни, одинаковые по признакам клиенты
могут поступить по-разному. Бонус синтетики: истинное p нам известно, поэтому
можно посчитать "потолок" — ROC-AUC идеальной модели (Bayes-optimal), и сравнивать
модели не только друг с другом, но и с теоретическим максимумом.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

# Коэффициенты логита: "истинная" модель поведения клиентов.
# Свободный член подобран так, чтобы доля оттока была ~25% (типично для телекома).
INTERCEPT = -4.5


def generate_synthetic_data(n_samples: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    """Генерирует синтетических клиентов телеком-оператора.

    Возвращает DataFrame с признаками, таргетом `churn` и истинной вероятностью
    оттока `churn_proba_true` (только для оценки, не признак!).
    """
    # Свой генератор вместо np.random.seed(): глобальный seed может сбить любой
    # другой код, вызывающий np.random. Объект Generator изолирован.
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 73, n_samples)  # месяцев с нами: 1..72
    monthly_charges = rng.uniform(20, 150, n_samples)
    # Раньше total_charges генерировался независимо (корреляция со стажем ~0).
    # В реальности это сумма всех платежей: ~ стаж x платёж. Множитель < 1 означает,
    # что раньше клиент платил меньше, то есть тариф с тех пор подорожал.
    past_price_factor = rng.uniform(0.75, 1.05, n_samples)
    total_charges = tenure * monthly_charges * past_price_factor

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_services": rng.integers(1, 8, n_samples),
            "has_internet": rng.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "has_phone": rng.choice([0, 1], n_samples, p=[0.4, 0.6]),
            "payment_method": rng.choice(
                ["bank_transfer", "credit_card", "paypal", "check"], n_samples
            ),
            "contract_type": rng.choice(
                ["month-to-month", "one_year", "two_year"], n_samples, p=[0.5, 0.3, 0.2]
            ),
            "age": rng.integers(18, 80, n_samples),
        }
    )

    # ========== ЛОГИКА ОТТОКА ==========
    # Насколько подорожал тариф: текущий платёж / средний платёж за всё время - 1.
    # Сырыми признаками это не выражается линейно, нужен признак-отношение
    # (charge_ratio в feature_engineering.py). Так feature engineering получает смысл.
    price_increase = monthly_charges / (total_charges / tenure) - 1

    logit = (
        INTERCEPT
        + 2.0 * (1 - tenure / 72)  # новые клиенты уходят чаще
        + 1.0 * (monthly_charges / 150)  # дорогой тариф
        + 1.5 * (df["contract_type"] == "month-to-month")  # нет обязательств
        - 0.5 * (df["contract_type"] == "two_year")
        + 0.6 * (df["num_services"] < 4)  # мало услуг -> слабая привязка
        + 4.0 * price_increase  # недавнее подорожание
        + 0.4 * (df["payment_method"] == "check")  # ручная оплата -> чаще забывают
        + 0.3 * (df["age"] < 30)
    )
    proba = 1 / (1 + np.exp(-logit))

    df[config.ORACLE_COLUMN] = proba
    df[config.TARGET_COLUMN] = rng.binomial(1, proba)  # случайный исход, а не порог

    print(f"Сгенерировано {n_samples:,} записей, доля оттока {df[config.TARGET_COLUMN].mean():.1%}")
    return df


def save_data(df: pd.DataFrame, filename: str) -> None:
    """Сохраняет сырые данные, чтобы не генерировать их каждый раз."""
    filepath = config.RAW_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")


def load_raw_data(filename: str = "churn.csv") -> pd.DataFrame:
    filepath = config.RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден. Сначала сгенерируйте данные.")
    return pd.read_csv(filepath)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Отделяет признаки от таргета и от истинной вероятности.

    ORACLE_COLUMN обязательно убираем из X: это сама вероятность оттока, с ней любая
    модель получит идеальный результат. Этот инвариант проверяется в tests/.
    """
    X = df.drop(columns=[config.TARGET_COLUMN, config.ORACLE_COLUMN])
    return X, df[config.TARGET_COLUMN], df[config.ORACLE_COLUMN]


def split_and_save_data(df: pd.DataFrame) -> None:
    """Делит данные на train/test (стратифицированно) и сохраняет.

    Test откладывается один раз и используется только в самом конце (evaluate.py).
    Выбор модели и гиперпараметров — по кросс-валидации на train (train.py).
    """
    X, y, oracle = split_features_target(df)
    X_train, X_test, y_train, y_test, _, oracle_test = train_test_split(
        X,
        y,
        oracle,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # одинаковая доля оттока в train и test
    )

    print(f"Train: {len(X_train):,} записей, отток {y_train.mean():.1%}")
    print(f"Test:  {len(X_test):,} записей, отток {y_test.mean():.1%}")

    out = config.PROCESSED_DATA_DIR
    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)
    oracle_test.to_csv(out / "oracle_test.csv", index=False)
    print(f"Данные сохранены в {out}")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    out = config.PROCESSED_DATA_DIR
    X_train = pd.read_csv(out / "X_train.csv")
    X_test = pd.read_csv(out / "X_test.csv")
    y_train = pd.read_csv(out / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(out / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test
