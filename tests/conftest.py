"""Общие настройки тестов: корень проекта в sys.path, маленький датасет."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import generate_synthetic_data  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic_data(n_samples=2_000, random_state=0)
