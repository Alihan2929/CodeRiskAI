"""Model, rapor ve ornek veri islemleri (FastAPI'den bagimsiz).

models/ ve reports/ klasorlerindeki mevcut dosyalari kullanir;
proje klasoru nereden calistirilirsa calistirilsin yollar dogru kalir.
"""

import csv
import math
import random
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "coderisk_model.joblib"
METADATA_PATH = ROOT / "models" / "coderisk_metadata.joblib"
REPORTS_DIR = ROOT / "reports"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

_cache = {}


def _load():
    """Model ve metadata dosyalarini bir kez yukler."""
    if not _cache:
        for path in (MODEL_PATH, METADATA_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Dosya bulunamadi: {path.name}")
        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["metadata"] = joblib.load(METADATA_PATH)
    return _cache["model"], _cache["metadata"]


def _read_csv(name):
    path = REPORTS_DIR / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_model_info():
    """Arayuzdeki 'Genel Bakis' ve 'Model' sayfalari icin ozet bilgi."""
    _, metadata = _load()

    final_rows = _read_csv("final_test_results.csv")
    final = {}
    if final_rows:
        final = {key: _num(value) for key, value in final_rows[0].items()}

    comparison = [
        {key: _num(value) for key, value in row.items()}
        for row in _read_csv("threshold_comparison.csv")
    ]

    importance = []
    for row in _read_csv("feature_importance.csv"):
        values = list(row.values())
        importance.append({"feature": values[0], "importance": _num(values[1])})
    importance.sort(key=lambda item: item["importance"] or 0, reverse=True)

    return {
        "model_name": metadata.get("model_name", "RandomForestClassifier"),
        "threshold": float(metadata["threshold"]),
        "features": list(metadata["feature_names"]),
        "final_results": final,
        "threshold_comparison": comparison,
        "feature_importance": importance,
    }


def predict(metrics):
    """21 metrikten kusur riskini tahmin eder (src/predict.py kullanilir)."""
    from src.predict import predict_defect

    model, metadata = _load()
    clean = {}
    for name, value in metrics.items():
        number = _num(value)
        if number is None:
            raise ValueError(f"'{name}' sayisal bir deger olmali.")
        clean[name] = number
    return predict_defect(clean, model, metadata)


def get_sample():
    """Test verisinden rastgele bir kayit dondurur (gercek etiketiyle)."""
    _, metadata = _load()
    if not TEST_PATH.exists():
        raise FileNotFoundError("data/processed/test.csv bulunamadi.")
    with TEST_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = random.choice(rows)
    metrics = {name: _num(row[name]) for name in metadata["feature_names"]}
    actual = str(row.get("defects", "")).strip().lower() in {"true", "1", "1.0"}
    return {"metrics": metrics, "actual_defect": actual}
