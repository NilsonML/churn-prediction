"""
evaluate.py - Финальная оценка обученных моделей на отложенном test.

Что считаем:
  ROC-AUC, PR-AUC  — качество ранжирования клиентов по риску.
  Brier score      — средняя квадратичная ошибка вероятностей: насколько им можно
                     верить как вероятностям (калибровка). Меньше — лучше.
  Precision@10%    — бизнес-метрика: если бюджет на удержание позволяет позвонить
                     10% клиентов с самым высоким риском, какая доля из них
                     действительно собиралась уйти.
  Oracle           — те же метрики для ИСТИННЫХ вероятностей, из которых сэмплирован
                     таргет. Это потолок: ни одна модель не может стабильно его превзойти.
"""

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import config
from src.data_processing import load_processed_data

matplotlib.use("Agg")  # рисуем в файлы без GUI: работает на сервере и в CI

TOP_SHARE = 0.10
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
ORACLE_COLOR = "#52514e"


def precision_at_top(y_true: np.ndarray, scores: np.ndarray, share: float = TOP_SHARE) -> float:
    """Доля реально ушедших среди `share` клиентов с самым высоким скором."""
    k = max(1, int(len(scores) * share))
    top = np.argsort(-scores)[:k]
    return float(np.asarray(y_true)[top].mean())


def compute_metrics(y_true, proba) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "precision_at_10pct": precision_at_top(y_true, proba),
    }


def load_models() -> dict:
    models = {}
    for path in sorted(config.MODELS_DIR.glob("*.pkl")):
        models[path.stem] = joblib.load(path)
    return models


def plot_curves(scores: dict[str, np.ndarray], y_test: pd.Series) -> None:
    """ROC- и PR-кривые всех моделей и оракула на одном рисунке."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, proba) in enumerate(scores.items()):
        is_oracle = name == "oracle"
        style = {
            "color": ORACLE_COLOR if is_oracle else COLORS[i % len(COLORS)],
            "linestyle": "--" if is_oracle else "-",
            "linewidth": 1.5 if is_oracle else 2,
        }
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, proba):.3f})", **style)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax_pr.plot(
            rec, prec, label=f"{name} ({average_precision_score(y_test, proba):.3f})", **style
        )

    ax_roc.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (AUC)")
    ax_pr.axhline(y_test.mean(), color="#c3c2b7", linewidth=1)
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (AP)")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="lower right" if ax is ax_roc else "upper right", frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def explain_with_shap(model, X_test: pd.DataFrame, name: str, n_samples: int = 1000) -> None:
    """SHAP-важность признаков для бустинга. Необязательный шаг: нужен пакет shap."""
    try:
        import shap
    except ImportError:
        print("SHAP пропущен: пакет shap не установлен")
        return
    try:
        # Всё, кроме последнего шага пайплайна, = превращение сырых данных в признаки
        X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=config.RANDOM_STATE)
        X_transformed = model[:-1].transform(X_sample)
        feature_names = model[:-1].get_feature_names_out()
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

        explainer = shap.TreeExplainer(model[-1])
        shap_values = explainer.shap_values(X_transformed)
        shap.summary_plot(shap_values, X_transformed, show=False, max_display=15)
        plt.title(f"SHAP: {name}")
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP сохранён: reports/shap_summary_{name}.png")
    except Exception as e:  # SHAP — диагностика, его сбой не должен ронять оценку
        print(f"SHAP для {name} не удался: {e}")


def to_markdown(df: pd.DataFrame) -> str:
    """Таблица для README без зависимости от пакета tabulate."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = [
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def evaluate_all() -> pd.DataFrame:
    config.ensure_dirs()
    _, X_test, _, y_test = load_processed_data()
    oracle = pd.read_csv(config.PROCESSED_DATA_DIR / "oracle_test.csv").squeeze("columns")

    models = load_models()
    if not models:
        raise RuntimeError("Нет обученных моделей. Сначала запустите: python main.py --train")

    scores = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
    scores["oracle"] = oracle.to_numpy()

    results = pd.DataFrame(
        [{"model": name, **compute_metrics(y_test, proba)} for name, proba in scores.items()]
    ).sort_values("roc_auc", ascending=False)
    results.to_csv(config.REPORTS_DIR / "test_metrics.csv", index=False)
    (config.REPORTS_DIR / "test_metrics.md").write_text(to_markdown(results))
    print("\nМетрики на test:\n" + results.round(4).to_string(index=False))

    plot_curves(scores, y_test)

    # SHAP для лучшего бустинга (у логрега важность видна прямо по коэффициентам)
    tree_models = [m for m in results["model"] if m not in ("oracle", "logreg")]
    if tree_models:
        explain_with_shap(models[tree_models[0]], X_test, tree_models[0])
    return results


if __name__ == "__main__":
    evaluate_all()
