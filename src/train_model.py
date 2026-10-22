
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


THRESHOLD = 0.40


def load_data():
    base_dir = Path("data/processed")

    train_data = pd.read_csv(base_dir / "train.csv")
    test_data = pd.read_csv(base_dir / "test.csv")

    X_train = train_data.drop(columns=["defects"])
    y_train = train_data["defects"].astype(bool)

    X_test = test_data.drop(columns=["defects"])
    y_test = test_data["defects"].astype(bool)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Egitim ve test sutunlari uyusmuyor."
        )

    print("=== VERI BILGILERI ===")
    print("Egitim kayitlari:", len(X_train))
    print("Test kayitlari:", len(X_test))
    print("Ozellik sayisi:", X_train.shape[1])
    print(
        "Egitim kusur orani:",
        round(y_train.mean() * 100, 2), "%"
    )
    print(
        "Test kusur orani:",
        round(y_test.mean() * 100, 2), "%"
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nModel egitiliyor...")
    model.fit(X_train, y_train)

    # Model ve model bilgilerini kaydet
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "coderisk_model.joblib"
    metadata_path = model_dir / "coderisk_metadata.joblib"

    metadata = {
        "feature_names": X_train.columns.tolist(),
        "threshold": THRESHOLD,
        "model_name": "RandomForestClassifier",
    }

    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)

    print("Model kaydedildi:", model_path)
    print("Model bilgileri kaydedildi:", metadata_path)

    # Test verisiyle tahmin
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities >= THRESHOLD

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\n=== SON TEST SONUCLARI ===")
    print("Tahmin esigi:", THRESHOLD)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\n=== CONFUSION MATRIX ===")
    print("Satirlar: Gercek, sutunlar: Tahmin")
    print(
        confusion_matrix(
            y_test, predictions, labels=[False, True]
        )
    )

    roc_auc = None

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(y_test, probabilities)
        print("\nROC-AUC:", round(roc_auc, 4))
    else:
        print(
            "\nROC-AUC hesaplanamadi: "
            "Testte tek sinif var."
        )

    print("\n=== SINIFLANDIRMA RAPORU ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[False, True],
            target_names=["Kusursuz", "Kusurlu"],
            zero_division=0,
        )
    )

    # Sonuçları CSV dosyasına kaydet
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }])

    results_path = output_dir / "final_test_results.csv"
    results.to_csv(results_path, index=False)

    print("\nSon test raporu kaydedildi:", results_path)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    evaluate_model(
        X_train, X_test, y_train, y_test
    )
