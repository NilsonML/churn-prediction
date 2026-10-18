"""
main.py - Главный скрипт для запуска всего пайплайна.
Последовательность действий:
1. Генерация данных и train/test split
2. Обучение моделей (feature engineering — шаг внутри каждого пайплайна)
3. Оценка и сравнение
"""

import argparse  # для парсинга аргументов командной строки

from src.data_processing import generate_synthetic_data, save_data, split_and_save_data
from src.train import main as train_main
from src.evaluate import evaluate_all

def main():
    """
    Основной пайплайн обучения и оценки моделей.
    
    Использование:
        python main.py --generate  # только сгенерировать данные
        python main.py --train     # только обучить модели
        python main.py --evaluate  # только оценить
        python main.py --all       # сделать всё
    """
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Churn Prediction Pipeline')
    parser.add_argument('--generate', action='store_true', help='Generate data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    
    args = parser.parse_args()
    
    # Если ничего не указано - запускаем всё
    if not any([args.generate, args.train, args.evaluate, args.all]):
        args.all = True
    
    print("\n" + "="*60)
    print("🏠 CHURN PREDICTION PIPELINE")
    print("="*60)
    
    # Шаг 1: Генерация данных
    if args.generate or args.all:
        print("\n📊 ШАГ 1: Генерация данных")
        df = generate_synthetic_data(n_samples=100_000)
        save_data(df, "churn.csv")
        split_and_save_data(df)
    
    # Шаг 2: Обучение моделей
    if args.train or args.all:
        print("\n" + "="*60)
        print("🤖 ШАГ 2: Обучение моделей")
        train_main()
    
    # Шаг 3: Оценка
    if args.evaluate or args.all:
        print("\n" + "="*60)
        print("📊 ШАГ 3: Оценка моделей")
        evaluate_all()
    
    print("\n" + "="*60)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЕН!")
    print("="*60)

if __name__ == "__main__":
    main()