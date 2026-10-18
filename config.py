"""
config.py - Централизованное хранение всех конфигураций проекта.
Здесь мы храним пути, гиперпараметры и настройки, чтобы легко менять их.
"""

import os
from pathlib import Path

# ========== ПУТИ ==========
# BASE_DIR - корневая папка проекта
BASE_DIR = Path(__file__).parent

# Пути к данным
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Пути к моделям
MODELS_DIR = BASE_DIR / "models"

# Создаем папки, если их нет
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========== НАСТРОЙКИ ДАННЫХ ==========
# Целевая переменная (что предсказываем)
TARGET_COLUMN = "churn"

# Списки колонок для модели — в src/feature_engineering.get_feature_columns()

# ========== ГИПЕРПАРАМЕТРЫ МОДЕЛЕЙ ==========
# CatBoost
CATBOOST_PARAMS = {
    "iterations" : 300,
    "learning_rate" : 0.05,
    "depth": 6,
    "verbose" : 50,
    "random_seed" : 42,
    # early_stopping_rounds убран: без eval_set остановке не на что смотреть,
    # параметр молча ничего не делал. Вернуть вместе с валидационной выборкой.
}

# LightGBM
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1
}

# XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 42,
    "verbosity": 0,  # у XGBClassifier параметр называется verbosity, а не verbose
}

# ========== НАСТРОЙКИ ОБУЧЕНИЯ ==========
TEST_SIZE = 0.2  # доля тестовой выборки
RANDOM_STATE = 42  # для воспроизводимости