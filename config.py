"""
config.py - Централизованное хранение всех конфигураций проекта.
Здесь мы храним пути, гиперпараметры и настройки, чтобы легко менять их.
"""

from pathlib import Path

# ========== ПУТИ ==========
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Обученные модели (*.pkl) — не коммитятся, их можно воспроизвести.
MODELS_DIR = BASE_DIR / "models"
# Метрики и графики — коммитятся: это результат проекта, на них ссылается README.
REPORTS_DIR = BASE_DIR / "reports"


def ensure_dirs() -> None:
    """Создаёт папки для артефактов.

    Раньше это делалось прямо при импорте config. Импорт с побочными эффектами —
    антипаттерн: любой `import config` (в тестах, в ноутбуке) создавал папки на диске.
    """
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# ========== НАСТРОЙКИ ДАННЫХ ==========
TARGET_COLUMN = "churn"

# Истинная вероятность оттока, из которой сэмплирован таргет. Известна только потому,
# что данные синтетические. НЕ признак (это была бы утечка ответа): используется
# лишь для оценки "потолка" — лучшего ROC-AUC, достижимого на этих данных.
ORACLE_COLUMN = "churn_proba_true"

# Исходные признаки
CATEGORICAL_COLUMNS = ["payment_method", "contract_type"]
BINARY_COLUMNS = ["has_internet", "has_phone"]
NUMERICAL_COLUMNS = ["tenure", "monthly_charges", "total_charges", "num_services", "age"]

# ========== ГИПЕРПАРАМЕТРЫ МОДЕЛЕЙ ==========
# Одинаковый бюджет (300 деревьев, lr 0.05) для всех бустингов, чтобы сравнение было честным.
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": 0,
    # early_stopping_rounds убран: без eval_set он ничего не делает
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 42,
    "verbosity": 0,  # у XGBClassifier параметр называется verbosity, а не verbose
}

HIST_GB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "random_state": 42,
}

# ========== НАСТРОЙКИ ОБУЧЕНИЯ ==========
N_SAMPLES = 100_000
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
