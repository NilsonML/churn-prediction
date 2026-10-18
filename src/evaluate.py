"""
evaluate.py - Оценка качества моделей и их сравнение.
Включает метрики, визуализацию и интерпретацию с SHAP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
import joblib
import shap
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к корневой папке
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.feature_engineering import create_features, get_feature_columns


def load_models():
    """
    Загружает все обученные модели.
    
    Returns:
        dict: словарь {имя_модели: объект_модели}
    """
    models = {}
    for model_name in ['catboost', 'lightgbm', 'xgboost']:
        model_path = config.MODELS_DIR / f'{model_name}_model.pkl'
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
            print(f"✅ Загружена модель {model_name}")
        else:
            print(f"⚠️ Модель {model_name} не найдена")
    return models


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Рассчитывает все основные метрики классификации.
    
    Args:
        y_true: истинные значения
        y_pred: предсказанные классы
        y_pred_proba: вероятности предсказаний
    
    Returns:
        dict: словарь со всеми метриками
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Строит и сохраняет матрицу ошибок.
    
    Матрица ошибок показывает:
        - TN: правильно предсказали, что клиент не уйдет
        - FP: ошибочно предсказали уход (ложная тревога)
        - FN: пропустили уход (самая дорогая ошибка)
        - TP: правильно предсказали уход
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Not Churn', 'Churn'],
        yticklabels=['Not Churn', 'Churn']
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    # Сохраняем
    save_path = config.MODELS_DIR / f'confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Матрица ошибок сохранена для {model_name}")


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Строит ROC-кривые для всех моделей на одном графике.
    
    ROC-кривая показывает:
        - По оси X: FPR (сколько ложных тревог)
        - По оси Y: TPR (сколько уходов нашли)
        - Чем выше кривая и больше AUC, тем лучше модель
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        # Получаем вероятности
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Рассчитываем ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Рисуем кривую
        plt.plot(
            fpr, tpr, 
            label=f'{name} (AUC = {auc:.4f})', 
            linewidth=2
        )
    
    # Диагональная линия - случайное угадывание
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1)
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем
    save_path = config.MODELS_DIR / 'roc_curves_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 ROC-кривые сохранены")


def explain_with_shap(model, X_test_original, model_name, preprocessor):
    """
    Исправленная версия SHAP для интерпретации модели.
    
    КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
        1. Применяем препроцессинг к данным (StandardScaler + OneHotEncoder)
        2. Получаем имена признаков после трансформации
        3. Передаем в SHAP уже преобразованные данные
    
    Args:
        model: обученная модель (Pipeline)
        X_test_original: исходные данные (без предобработки)
        model_name: имя модели для сохранения
        preprocessor: объект препроцессора из пайплайна
    """
    try:
        print(f"\n🔍 Анализ SHAP для {model_name}...")
        
        # ========== ШАГ 1: Применяем препроцессинг ==========
        # Это самое важное исправление!
        # SHAP должен получать те же данные, что и модель
        
        print("   → Применяем препроцессинг к данным...")
        X_test_transformed = preprocessor.transform(X_test_original)
        
        # ========== ШАГ 2: Получаем имена признаков после трансформации ==========
        # OneHotEncoder создает новые имена: payment_method_credit_card, и т.д.
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # fallback для старых версий sklearn
            feature_names = [
                f"feature_{i}" for i in range(X_test_transformed.shape[1])
            ]
        
        # Создаем DataFrame с правильными именами
        X_test_transformed_df = pd.DataFrame(
            X_test_transformed,
            columns=feature_names,
            index=X_test_original.index
        )
        
        print(f"   → После препроцессинга: {X_test_transformed_df.shape[1]} признаков")
        
        # ========== ШАГ 3: Извлекаем классификатор из пайплайна ==========
        classifier = model.named_steps['classifier']
        
        # ========== ШАГ 4: Создаем SHAP объяснитель ==========
        print("   → Создаем SHAP объяснитель...")
        
        # Выбираем подходящий объяснитель для модели
        if 'catboost' in model_name.lower():
            explainer = shap.TreeExplainer(classifier)
        elif 'lightgbm' in model_name.lower():
            explainer = shap.TreeExplainer(classifier)
        elif 'xgboost' in model_name.lower():
            # Для XGBoost нужно преобразовать данные
            explainer = shap.TreeExplainer(classifier)
        else:
            print(f"⚠️ Неподдерживаемая модель для SHAP: {model_name}")
            return
        
        # ========== ШАГ 5: Рассчитываем SHAP значения ==========
        # Берем подвыборку для скорости (если данных много)
        n_samples_for_shap = min(1000, len(X_test_transformed_df))
        X_sample = X_test_transformed_df.sample(
            n=n_samples_for_shap,
            random_state=42
        )
        
        print(f"   → Рассчитываем SHAP для {n_samples_for_shap} наблюдений...")
        shap_values = explainer.shap_values(X_sample)
        
        # ========== ШАГ 6: Визуализация ==========
        
        # 6.1. Summary Plot (главный график)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            show=False,
            max_display=15,  # показываем топ-15 признаков
            plot_size=(12, 10)
        )
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = config.MODELS_DIR / f'shap_summary_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ SHAP summary сохранен: shap_summary_{model_name}.png")
        
        # 6.2. Bar Plot (ранжирование важности)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            show=False,
            plot_type="bar",
            max_display=15
        )
        plt.title(f'SHAP Feature Importance (Bar) - {model_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = config.MODELS_DIR / f'shap_bar_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ SHAP bar сохранен: shap_bar_{model_name}.png")
        
        # 6.3. Сохраняем SHAP значения для дальнейшего анализа
        shap_df = pd.DataFrame(
            shap_values,
            columns=X_sample.columns,
            index=X_sample.index
        )
        shap_df.to_csv(
            config.MODELS_DIR / f'shap_values_{model_name}.csv'
        )
        print(f"   ✅ SHAP значения сохранены в shap_values_{model_name}.csv")
        
        # 6.4. Топ-10 важнейших признаков (текстовый вывод)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        print(f"\n   🏆 Топ-10 важнейших признаков ({model_name}):")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
    except Exception as e:
        print(f"⚠️ Ошибка SHAP для {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def evaluate_all():
    """
    Главная функция для оценки всех моделей.
    """
    print("\n" + "="*60)
    print("📊 ОЦЕНКА МОДЕЛЕЙ")
    print("="*60)
    
    # ========== 1. Загружаем данные ==========
    print("\n📂 Загрузка данных...")
    X_test_raw = pd.read_csv(config.PROCESSED_DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / 'y_test.csv').squeeze()
    print(f"   → Тестовых данных: {len(X_test_raw)} записей")
    
    # ========== 2. Создаем признаки ==========
    X_test = create_features(X_test_raw)
    print(f"   → После feature engineering: {X_test.shape[1]} признаков")
    
    # Определяем колонки для препроцессинга
    feature_cols = get_feature_columns(X_test)
    numerical_cols = feature_cols['numerical']
    categorical_cols = feature_cols['categorical']
    
    # ========== 3. Загружаем модели ==========
    print("\n📂 Загрузка моделей...")
    models = load_models()
    
    if not models:
        print("❌ Нет моделей для оценки. Сначала обучите их.")
        return
    
    # ========== 4. Оцениваем каждую модель ==========
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"📈 {name.upper()}")
        print('='*50)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        
        # Выводим метрики
        print("\n📊 Метрики:")
        for metric, value in metrics.items():
            print(f"   {metric:>10}: {value:.4f}")
        
        # Матрица ошибок
        plot_confusion_matrix(y_test, y_pred, name)
        
        # Classification Report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # ========== SHAP АНАЛИЗ (ИСПРАВЛЕННАЯ ВЕРСИЯ) ==========
        # Получаем препроцессор из пайплайна
        preprocessor = model.named_steps['preprocessor']
        explain_with_shap(model, X_test, name, preprocessor)
    
    # ========== 5. ROC-кривые ==========
    plot_roc_curves(models, X_test, y_test)
    
    # ========== 6. Сравнительная таблица ==========
    print("\n" + "="*60)
    print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
    print("="*60)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # Сохраняем результаты
    results_df.to_csv(config.MODELS_DIR / 'model_comparison.csv')
    print("\n💾 Результаты сохранены в models/model_comparison.csv")
    
    # ========== 7. Выбираем лучшую модель ==========
    best_model = results_df['roc_auc'].idxmax()
    print(f"\n🏆 Лучшая модель по ROC-AUC: {best_model}")
    print(f"   ROC-AUC: {results_df.loc[best_model, 'roc_auc']:.4f}")
    
    # ========== 8. Дополнительный анализ ==========
    print("\n" + "="*60)
    print("📈 ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("="*60)
    
    # Анализ порогов классификации
    best_model_obj = models[best_model]
    y_pred_proba_best = best_model_obj.predict_proba(X_test)[:, 1]
    
    # Проверяем распределение вероятностей
    print(f"\n📊 Распределение вероятностей (лучшая модель):")
    print(f"   Min: {y_pred_proba_best.min():.4f}")
    print(f"   Max: {y_pred_proba_best.max():.4f}")
    print(f"   Mean: {y_pred_proba_best.mean():.4f}")
    print(f"   Median: {np.median(y_pred_proba_best):.4f}")
    
    # Группы риска
    risk_groups = pd.cut(
        y_pred_proba_best,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    print(f"\n🎯 Распределение по группам риска:")
    for group in ['Low Risk', 'Medium Risk', 'High Risk']:
        count = (risk_groups == group).sum()
        pct = count / len(risk_groups) * 100
        print(f"   {group:>12}: {count:>6,} ({pct:.1f}%)")
    
    return results, best_model


if __name__ == "__main__":
    evaluate_all()