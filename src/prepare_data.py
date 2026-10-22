
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def prepare_data():
    data_dir = Path("data/processed")

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "train.csv veya test.csv bulunamadi. "
            "Once veri hazirlama islemini tamamla."
        )

    # Mevcut egitim kumesini oku
    data = pd.read_csv(train_path)

    X = data.drop(columns=["defects"])
    y = data["defects"].astype(bool)

    # Ayni metriklere sahip kayitlari ayni grupta tut
    groups = pd.util.hash_pandas_object(X, index=False)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, validation_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_validation = X.iloc[validation_idx]

    y_train = y.iloc[train_idx]
    y_validation = y.iloc[validation_idx]

    # Dosyalari olustur
    train_data = X_train.copy()
    train_data["defects"] = y_train

    validation_data = X_validation.copy()
    validation_data["defects"] = y_validation

    # Sadece egitim dosyasi ve yeni dogrulama dosyasi yazilir.
    # test.csv bu dosyada degistirilmez.
    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(
        data_dir / "validation.csv",
        index=False,
    )

    train_groups = set(groups.iloc[train_idx])
    validation_groups = set(groups.iloc[validation_idx])

    print("=== EGITIM / DOGRULAMA AYRIMI ===")
    print("Egitim kayit sayisi:", len(train_data))
    print("Dogrulama kayit sayisi:", len(validation_data))
    print(
        "Egitim kusur orani:",
        round(y_train.mean() * 100, 2),
        "%",
    )
    print(
        "Dogrulama kusur orani:",
        round(y_validation.mean() * 100, 2),
        "%",
    )
    print(
        "Ortak metrik grubu sayisi:",
        len(train_groups & validation_groups),
    )

    print("\nEgitim dosyasi guncellendi:", train_path)
    print("Dogrulama dosyasi kaydedildi:", data_dir / "validation.csv")
    print("Test dosyasina dokunulmadi:", test_path)


if __name__ == "__main__":
    prepare_data()
