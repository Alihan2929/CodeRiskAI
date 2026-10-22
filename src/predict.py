
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/coderisk_model.joblib")
METADATA_PATH = Path("models/coderisk_metadata.joblib")

INPUT_PATH = Path("data/raw/jm1.csv")
OUTPUT_PATH = Path("reports/prediction_results.csv")


def load_model():
    """Eğitilmiş modeli ve model bilgilerini yükler."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyasi bulunamadi: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model bilgileri bulunamadi: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    return model, metadata


def get_risk_level(probability):
    """Kusur olasılığına göre risk seviyesini belirler."""
    if probability < 0.20:
        return "Dusuk"
    elif probability < 0.40:
        return "Orta"
    elif probability < 0.70:
        return "Yuksek"
    else:
        return "Cok yuksek"


def predict_defect(metrics, model=None, metadata=None):
    """Tek bir kaydın kusur riskini tahmin eder."""
    if model is None or metadata is None:
        model, metadata = load_model()

    feature_names = metadata["feature_names"]
    threshold = float(metadata["threshold"])

    missing_features = [
        name for name in feature_names
        if name not in metrics
    ]

    extra_features = [
        name for name in metrics
        if name not in feature_names
    ]

    if missing_features or extra_features:
        raise ValueError(
            f"Eksik ozellikler: {missing_features}\n"
            f"Fazladan ozellikler: {extra_features}"
        )

    input_data = pd.DataFrame(
        [[metrics[name] for name in feature_names]],
        columns=feature_names,
    )

    probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    prediction = probability >= threshold

    return {
        "defect_probability": round(probability * 100, 2),
        "prediction": "Kusurlu" if prediction else "Kusursuz",
        "risk_level": get_risk_level(probability),
        "threshold": threshold,
    }


def predict_csv(input_path, output_path):
    """CSV dosyasındaki tüm kayıtları analiz edip sonuçları kaydeder."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Girdi dosyasi bulunamadi: {input_path}"
        )

    model, metadata = load_model()
    feature_names = metadata["feature_names"]
    threshold = float(metadata["threshold"])

    data = pd.read_csv(input_path)

    # Gerçek etiket varsa tahminde kullanma.
    # Yalnızca özellik sütunlarını modele gönder.
    missing_features = [
        name for name in feature_names
        if name not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"CSV dosyasinda eksik ozellikler var: "
            f"{missing_features}"
        )

    X = data[feature_names].copy()

    if X.isnull().any().any():
        raise ValueError(
            "Girdi verisinde eksik degerler var. "
            "Tahminden once bu degerleri incelemelisin."
        )

    probabilities = model.predict_proba(X)[:, 1]
    predictions = probabilities >= threshold

    results = pd.DataFrame({
        "record_index": data.index,
        "defect_probability": (probabilities * 100).round(2),
        "prediction": [
            "Kusurlu" if value else "Kusursuz"
            for value in predictions
        ],
        "risk_level": [
            get_risk_level(float(value))
            for value in probabilities
        ],
        "threshold": threshold,
    })

    # Kaynak dosyada gerçek etiket varsa yalnızca
    # sonuçları sonradan değerlendirmek için ekle.
    if "defects" in data.columns:
        results["actual_defects"] = data["defects"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n=== CODERISK AI TOPLU TAHMIN SONUCU ===")
    print("Analiz edilen kayit:", len(results))
    print(
        "Kusurlu tahmini:",
        int((results["prediction"] == "Kusurlu").sum())
    )
    print(
        "Kusursuz tahmini:",
        int((results["prediction"] == "Kusursuz").sum())
    )
    print("\nRisk seviyelerinin dagilimi:")
    print(results["risk_level"].value_counts().to_string())
    print(f"\nSonuclar kaydedildi: {output_path}")


if __name__ == "__main__":
    predict_csv(INPUT_PATH, OUTPUT_PATH)
