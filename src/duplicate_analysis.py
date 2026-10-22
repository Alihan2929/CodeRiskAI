import pandas as pd


def analyze_duplicates(file_path):
    data = pd.read_csv(file_path)

    # Hedef etiket haricindeki metrikler
    features = data.drop(columns=["defects"])

    duplicate_mask = data.duplicated(keep=False)
    duplicate_rows = data[duplicate_mask]

    print("=== TEKRAR ANALIZI ===")
    print("Tekrar grubuna dahil satir:", len(duplicate_rows))

    print("\nTekrarli satirlardaki defects dagilimi:")
    print(duplicate_rows["defects"].value_counts())

    print("\nTekrarli satirlarda farkli tam kayit sayisi:")
    print(duplicate_rows.drop_duplicates().shape[0])

    print("\nBenzersiz metrik kombinasyonu sayisi:")
    print(features.drop_duplicates().shape[0])

    print("\nEtiketi farkli olan ayni metrikler:")

    grouped = data.groupby(
        list(features.columns),
        dropna=False
    )["defects"].nunique()

    print("Ayni metriklerle birden fazla etiket:", (grouped > 1).sum())


if __name__ == "__main__":
    analyze_duplicates("data/raw/jm1.csv")