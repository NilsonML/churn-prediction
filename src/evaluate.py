"""
evaluate.py - Оценка качества моделей и их сравнение.
Метрики, матрицы ошибок, ROC-кривые и интерпретация через SHAP.

Запуск отдельно:  python -m src.evaluate   (из корня проекта)
"""

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, без окон — работает и на сервере
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

import config  # noqa: E402
from src.models import MODEL_FACTORIES  # noqa: E402


def load_models() -> dict:
    """Загружает все обученные пайплайны, которые есть на диске."""
    models = {}
    for name in MODEL_FACTORIES:
        path = config.MODELS_DIR / f"{name}_model.pkl"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"✅ Загружена модель {name}")
        else:
            print(f"⚠️ Модель {name} не найдена — пропускаем")
    return models


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """FN (пропустили уход) обычно дороже FP (зря предложили скидку)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Churn", "Churn"], yticklabels=["Not Churn", "Churn"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close()


def plot_roc_curves(probas: dict, y_test) -> None:
    plt.figure(figsize=(7, 6))
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "roc_curves_comparison.png", dpi=150)
    plt.close()


def _positive_class_shap(shap_values) -> np.ndarray:
    """Разные версии SHAP/моделей возвращают значения в разной форме:
    список [класс 0, класс 1], массив (n, features, 2) или (n, features).
    Приводим к (n, features) для положительного класса."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    shap_values = np.asarray(shap_values)
    return shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values


def explain_with_shap(model, X_test: pd.DataFrame, model_name: str, n_samples: int = 1000) -> None:
    """SHAP: какие признаки сильнее всего влияют на предсказание.

    SHAP должен видеть те же данные, что и модель: прогоняем X через все
    шаги пайплайна, кроме последнего (model[:-1]), и берём имена колонок
    у ColumnTransformer.
    """
    try:
        import shap  # внутри try: без SHAP оценка всё равно должна завершиться

        X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=config.RANDOM_STATE)
        X_transformed = pd.DataFrame(
            model[:-1].transform(X_sample),
            columns=model.named_steps["preprocessor"].get_feature_names_out(),
            index=X_sample.index,
        )
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        values = _positive_class_shap(explainer.shap_values(X_transformed))

        for plot_type, suffix in ((None, "summary"), ("bar", "bar")):
            shap.summary_plot(values, X_transformed, show=False, max_display=15, plot_type=plot_type)
            plt.title(f"SHAP — {model_name}")
            plt.tight_layout()
            plt.savefig(config.MODELS_DIR / f"shap_{suffix}_{model_name}.png", dpi=150, bbox_inches="tight")
            plt.close()

        importance = (pd.Series(np.abs(values).mean(axis=0), index=X_transformed.columns)
                      .sort_values(ascending=False))
        importance.to_csv(config.MODELS_DIR / f"shap_importance_{model_name}.csv", header=["mean_abs_shap"])
        print(f"   🏆 Топ-5 признаков ({model_name}): {', '.join(importance.index[:5])}")
    except Exception as e:  # SHAP — дополнительный анализ, его сбой не должен ронять оценку
        print(f"⚠️ SHAP для {model_name} не посчитан: {e}")


def evaluate_all():
    print("\n" + "=" * 60)
    print("📊 ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 60)

    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    models = load_models()
    if not models:
        print("❌ Нет обученных моделей. Сначала запусти: python main.py --train")
        return None

    results, probas = {}, {}
    for name, model in models.items():
        print(f"\n📈 {name.upper()}")
        # Пайплайн принимает СЫРЫЕ признаки — feature engineering внутри.
        y_pred = model.predict(X_test)
        probas[name] = model.predict_proba(X_test)[:, 1]
        results[name] = calculate_metrics(y_test, y_pred, probas[name])
        print(classification_report(y_test, y_pred, digits=4))
        plot_confusion_matrix(y_test, y_pred, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explain_with_shap(model, X_test, name)

    plot_roc_curves(probas, y_test)

    results_df = pd.DataFrame(results).T.round(4)
    results_df.to_csv(config.MODELS_DIR / "model_comparison.csv")
    print("\n📊 Сравнение моделей:\n", results_df)

    best = results_df["roc_auc"].idxmax()
    print(f"\n🏆 Лучшая модель по ROC-AUC: {best} ({results_df.loc[best, 'roc_auc']:.4f})")

    # Группы риска: удобный формат для бизнеса ("кому звонить в первую очередь")
    risk = pd.cut(probas[best], bins=[0, 0.3, 0.7, 1.0], include_lowest=True,
                  labels=["Low Risk", "Medium Risk", "High Risk"])
    print("\n🎯 Группы риска (лучшая модель):")
    print(pd.Series(risk).value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    return results_df, best


if __name__ == "__main__":
    evaluate_all()
