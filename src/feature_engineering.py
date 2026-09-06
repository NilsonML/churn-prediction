"""
feature_engineering.py - Создание новых признаков из существующих.
Это один из самых важных этапов в ML - качественные признаки дают 
больше, чем сложная модель.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает новые признаки на основе существующих.
    
    Аргументы:
        df: DataFrame с исходными признаками
    
    Возвращает:
        DataFrame с добавленными признаками
    
    Логика создания признаков:
        1. Финансовые метрики
        2. Демографические группы
        3. Взаимодействия признаков
        4. Бинарные флаги для крайних значений
    """
    print("\n🔧 Создаем новые признаки...")
    df = df.copy()  # чтобы не менять оригинал
    
    # ========== 1. ФИНАНСОВЫЕ МЕТРИКИ ==========
    # Среднемесячный платеж за всё время
    # total_charges / tenure - показывает, сколько в среднем платит клиент
    # Если tenure = 0 (не бывает, т.к. у нас от 1), но на всякий случай
    df["avg_monthly_charge"] = df["total_charges"] / df["tenure"].clip(lower=1)
    
    # Отношение текущего платежа к среднему
    # > 1 значит, что платеж вырос, < 1 - упал
    df["charge_ratio"] = df["monthly_charges"] / df["avg_monthly_charge"]
    
    # ========== 2. ДЕМОГРАФИЧЕСКИЕ ГРУППЫ ==========
    # Молодые клиенты (до 30) часто более мобильны
    df["is_young"] = (df["age"] < 30).astype(int)
    
    # Пожилые клиенты (60+) часто более лояльны
    df["is_senior"] = (df["age"] > 60).astype(int)
    
    # Средний возраст (30-60) - референсная группа
    df["is_middle_age"] = ((df["age"]) & (df["age"] <= 60)).astype(int)
    
    # ========== 3. БИНАРНЫЕ ФЛАГИ ==========
    # Клиенты с высокими платежами (топ 25%)
    high_charge_threshold = df["monthly_charges"].quantile(0.75)
    df["high_charges"] = (df["monthly_charges"] > high_charge_threshold).astype(int)
    
    # Долгоживущие клиенты (топ 25% по стажу)
    long_tenure_threshold = df["tenure"].quantile(0.75)
    df["long_tenure"] = (df["tenure"] > long_tenure_threshold).astype(int)
    
    # ========== 4. ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ ==========
    # Клиенты с интернетом и высокими платежами
    df["internet_high_charge"] = (df["has_internet"] & df["high_charges"]).astype(int)
    
    # Молодые с помесячным контрактом (рискованная группа)
    df["young_monthly"] = (df["is_young"] &
                           (df["contract_type"] == "month-to-month")).astype(int)
    
    # ========== 5. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (преобразуем для моделей) ==========
    # Некоторые модели (LightGBM, XGBoost) лучше работают с категориальными
    # кодированными как числа, поэтому создаем числовые версии
    
    # Contract type: month-to-month = 0, one_year = 1, two_year = 2
    contract_map = {"month-to-month": 0, "one_year": 1, "two_year": 2}
    df["contract_type_num"] = df["contract_type"].map(contract_map)
    
    # Payment method: банк = 0, кредитка = 1, PayPal = 2, чек = 3
    payment_map = {"bank_transfer": 0, "credit_card": 1, "paypal": 2, "check": 3}
    df["payment_method_num"] = df["payment_method"].map(payment_map)
    
    print(f"✅ Добавлено {df.shape[1] - 8} новых признаков")
    print(f"📊 Общее количество признаков: {df.shape[1]}")
    
    return df

def get_feature_columns(df: pd.DataFrame) -> dict:
    """
    Определяет типы колонок после feature engineering.
    
    Аргументы:
        df: DataFrame с признаками
    
    Возвращает:
        dict с категориями колонок
    """
    # Все колонки кроме целевой
    all_columns = [col for col in df.columns if col != config.TARGET_COLUMN]
    
    # Категориальные колонки (включая новые)
    categorical = [
        "payment_method", 
        "contract_type",
        "is_young",
        "is_senior", 
        "is_middle_age",
        "high_charges",
        "long_tenure",
        "internet_high_charge",
        "young_monthly"
    ]
    
    # Числовые колонки
    numerical = [
        "tenure", 
        "monthly_charges", 
        "total_charges", 
        "num_services",
        "has_internet",
        "has_phone",
        "age",
        "avg_monthly_charge",
        "charge_ratio",
        "contract_type_num",
        "payment_method_num"
    ]
    
    return {
        "all": all_columns,
        "categorical": [col for col in categorical if col in df.columns],
        "numerical": [col for col in numerical if col in df.columns]
    }