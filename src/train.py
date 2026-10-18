"""
train.py - Обучение моделей машинного обучения.
Здесь мы создаем пайплайны с предобработкой и обучаем три модели.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config
from src.feature_engineering import create_features, get_feature_columns

def create_preprocessor(numerical_cols: list, categorical_cols: list):
    """
    Создает препроцессор для трансформации признаков.
    
    Аргументы:
        numerical_cols: список числовых колонок
        categorical_cols: список категориальных колонок
    
    Возвращает:
        ColumnTransformer для предобработки
    
    Что делаем:
        1. Числовые признаки → StandardScaler (нормализация)
        2. Категориальные признаки → One-Hot Encoding
    
    Почему:
        - Стандартизация нужна для моделей с градиентным спуском
        - One-Hot Encoding преобразует категории в бинарные признаки
    """
    
    # Для числовых: центрируем и масштабируем
    # (x - mean) / std → среднее 0, дисперсия 1
    numerical_transformer = StandardScaler()
    
    # Для категориальных: One-Hot Encoding
    # drop='first' - убираем первый столбец для избежания мультиколлинеарности
    categorical_transformer = OneHotEncoder(
        drop="first",
        sparse_output=False, # возвращаем плотный массив (не разреженный)
        handle_unknown="ignore" # если в тесте новые категории - игнорируем
    )
    
    # Объединяем трансформеры
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    return preprocessor

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   numerical_cols: list, categorical_cols: list) -> Pipeline:
    
    """
    Обучает модель CatBoost.
    
    CatBoost - лучший выбор для табличных данных, потому что:
    1. Автоматически обрабатывает категориальные признаки
    2. Устойчив к выбросам
    3. Дает хорошие результаты "из коробки"
    """
    print("\n" + "="*50)
    print("🚀 Обучаем CatBoost...")
    print("="*50)
    
    # Создаем препроцессор
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    # Создаем пайплайн: предобработка → модель
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', cb.CatBoostClassifier(**config.CATBOOST_PARAMS))
    ])
    
    # Обучаем
    model.fit(X_train, y_train)
    
    # Сохраняем
    joblib.dump(model, config.MODELS_DIR / 'catboost_model.pkl')
    print("💾 CatBoost сохранен")
    
    return model

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   numerical_cols: list, categorical_cols: list) -> Pipeline:
    """
    Обучает модель LightGBM.
    
    LightGBM - быстрый и эффективный, особенно на больших данных.
    Использует гистограммный подход для ускорения.
    """
    print("\n" + "="*50)
    print("🚀 Обучаем LightGBM...")
    print("="*50)
    
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS))
    ])
    
    model.fit(X_train, y_train)
    joblib.dump(model, config.MODELS_DIR / 'lightgbm_model.pkl')
    print("💾 LightGBM сохранен")
    
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  numerical_cols: list, categorical_cols: list) -> Pipeline:
    """
    Обучает модель XGBoost.
    
    XGBoost - классика, отлично работает, поддерживает регуляризацию.
    """
    print("\n" + "="*50)
    print("🚀 Обучаем XGBoost...")
    print("="*50)
    
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**config.XGBOOST_PARAMS))
    ])
    
    model.fit(X_train, y_train)
    joblib.dump(model, config.MODELS_DIR / 'xgboost_model.pkl')
    print("💾 XGBoost сохранен")
    
    return model


def load_processed_data() -> tuple:
    """
    Загружает обработанные данные из папки processed.
    """
    X_train = pd.read_csv(config.PROCESSED_DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / 'X_test.csv')
    y_train = pd.read_csv(config.PROCESSED_DATA_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / 'y_test.csv').squeeze()
    
    return X_train, X_test, y_train, y_test

def main():
    """
    Главная функция для обучения всех моделей.
    """
    print("\n" + "="*60)
    print("🏋️  НАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("="*60)
    
    # 1. Загружаем обработанные данные
    X_train_raw, X_test_raw, y_train, y_test = load_processed_data()
    
    # 2. Создаем новые признаки
    X_train = create_features(X_train_raw)
    X_test = create_features(X_test_raw)
    
    # 3. Определяем типы колонок
    feature_cols = get_feature_columns(X_train)
    numerical_cols = feature_cols['numerical']
    categorical_cols = feature_cols['categorical']
    
    print(f"\n📊 Признаки для обучения:")
    print(f"   - Числовые: {len(numerical_cols)}")
    print(f"   - Категориальные: {len(categorical_cols)}")
    print(f"   - Всего: {len(feature_cols['all'])}")
    
    # 4. Обучаем все три модели
    models = {
        'CatBoost': train_catboost(X_train, y_train, numerical_cols, categorical_cols),
        'LightGBM': train_lightgbm(X_train, y_train, numerical_cols, categorical_cols),
        'XGBoost': train_xgboost(X_train, y_train, numerical_cols, categorical_cols)
    }
    
    print("\n" + "="*60)
    print("✅ ВСЕ МОДЕЛИ ОБУЧЕНЫ И СОХРАНЕНЫ!")
    print("="*60)
    
    return models

if __name__ == "__main__":
    main()