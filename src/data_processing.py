"""
data_processing.py - Отвечает за загрузку данных и их предобработку.
Включает в себя:
1. Генерацию синтетических данных
2. Разделение на признаки (X) и целевую переменную (y)
3. Создание train/test сплита
4. Сохранение обработанных данных
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Добавляем путь к корневой папке для импорта config
sys.path.append(str(Path(__file__).parent.parent))
import config

def generate_synthetic_data(n_samples: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетические данные о клиентах.
    
    Аргументы:
        n_samples: количество записей
        random_state: seed для воспроизводимости
    
    Возвращает:
        pd.DataFrame с признаками и целевой переменной
    
    Логика генерации:
        - Создаем реалистичные признаки клиентов (стаж, тарифы, услуги)
        - Добавляем логику оттока (чем меньше стаж, выше платежи, 
          помесячный контракт - тем выше вероятность уйти)
    """
    
    np.random.seed(random_state)
    
    print(f"📊 Генерируем {n_samples:,} записей...")
    
    # Создаем признаки
    data = {
        # tenure - сколько месяцев клиент с нами (от 1 до 72 месяцев = 6 лет)
        "tenure": np.random.randint(1,72,n_samples),
        
        # monthly_charges - ежемесячный платеж (от 20 до 150 долларов)
        "monthly_charges": np.random.uniform(20,150,n_samples),
        
        # total_charges - общая сумма платежей за всё время
        "total_charges": np.random.uniform(200, 8000, n_samples),
        
        # num_services - количество подключенных услуг (от 1 до 7)
        "num_services" : np.random.randint(1,8,n_samples),
        
        # бинарные признаки: есть ли интернет/телефон
        "has_internet": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "has_phone": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        
        # категориальные признаки
        "payment_method" : np.random.choice(
            ["bank_transfer", "credit_card", "paypal", "check"],
            n_samples
        ),
        "contract_type": np.random.choice(
            ["month-to-month", "one_year", "two_year"],
            n_samples,
            p=[0.5,0.3,0.2] # 50% клиентов на помесячном контракте
        ),
        "age": np.random.randint(18,80,n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # ========== ЛОГИКА ОТТОКА ==========
    # Мы моделируем вероятность ухода клиента на основе его характеристик
    # Это имитация реального поведения
    
    # 1. Клиенты с меньшим стажем чаще уходят (0.3 * (1 - tenure/72))
    # tenure=0 → 0.3, tenure=72 → 0
    tenure_factor = 0.3 * (1-df["tenure"] / 72)
    
    # 2. Дорогие тарифы увеличивают отток (0.2 * monthly_charges/150)
    price_factor = 0.2 * (df["monthly_charges"] / 150)
    
    # 3. Помесячный контракт - высокий риск оттока
    contract_factor = 0.3 * (df["contract_type"] == "month-to-month").astype(int)
    
    # 4. Мало услуг - клиент не заинтересован
    services_factor = 0.1 * (df["num_services"] < 4).astype(int)
    
    # 5. Случайный шум (реальные данные всегда зашумлены)
    noise = 0.1 * np.random.random(n_samples)
    
    # Суммируем все факторы
    churn_prob = tenure_factor + price_factor + contract_factor + services_factor + noise
    
    # Если вероятность > 50%, клиент уходит
    df[config.TARGET_COLUMN] = (churn_prob > 0.5).astype(int)
    
    print(f"✅ Сгенерировано {n_samples:,} записей")
    print(f"📈 Доля оттока: {df[config.TARGET_COLUMN].mean():.2%}")
    print(f"📊 Количество признаков: {df.shape[1] - 1} (без целевой)")
    
    return df

def save_data(df: pd.DataFrame, filename: str) -> None:
    """
    Сохраняет DataFrame в CSV.
    
    Аргументы:
        df: данные для сохранения
        filename: имя файла
    
    Зачем: 
        Чтобы сохранить сырые данные и не генерировать их каждый раз
    """
    
    filepath = config.RAW_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"💾 Данные сохранены в {filepath}")
    
def load_raw_data(filename: str = "churn.csv") -> pd.DataFrame:
    """
    Загружает сырые данные из CSV.
    
    Аргументы:
        filename: имя файла
    
    Возвращает:
        pd.DataFrame с данными
    """
    filepath = config.RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден. Сначала сгенерируйте данные.")
    return pd.read_csv(filepath)
    
def split_and_save_data(df: pd.DataFrame) -> tuple:
    """
    Разделяет данные на train/test и сохраняет их.
    
    Аргументы:
        df: полный датасет
    
    Возвращает:
        tuple: (X_train, X_test, y_train, y_test) - все в pandas
    
    Почему сохраняем:
        - Чтобы не делать сплит каждый раз заново
        - Для воспроизводимости экспериментов
        - Для удобства работы в Jupyter
    """
    print("\n🔄 Разделяем данные на train/test...")
    
    # Отделяем признаки от целевой переменной
    X = df.drop(config.TARGET_COLUMN, axis=1)
    y = df[config.TARGET_COLUMN]
    
    # Разделяем с сохранением пропорций классов (stratify)
    # stratify=y - гарантирует, что в обеих выборках будет одинаковая доля оттока
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y, # <- важно для несбалансированных данных
    )
    
    print(f"📊 Обучающая выборка: {len(X_train):,} записей")
    print(f"📊 Тестовая выборка: {len(X_test):,} записей")
    print(f"📈 Отток в train: {y_train.mean():.2%}")
    print(f"📈 Отток в test: {y_test.mean():.2%}")
    
    # Сохраняем обработанные данные
    X_train.to_csv(config.PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(config.PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(config.PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(config.PROCESSED_DATA_DIR / "y_test.csv", index=False)
    
    print(f"💾 Данные сохранены в {config.PROCESSED_DATA_DIR}")
    
    return X_train, X_test, y_train, y_test