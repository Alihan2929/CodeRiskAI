from pathlib import Path
import pandas as pd


def check_data_quality(file_path):
    data = pd.read_csv(file_path)

    print("=== VERI KALITESI ANALIZI ===")

    print("\nVeri tipleri:")
    print(data.dtypes)

    print("\nTekrarlanan satir sayisi:")
    print(data.duplicated().sum())

    print("\nSonsuz deger sayisi:")
    numeric_data = data.select_dtypes(include="number")
    print(numeric_data.isin([float("inf"), float("-inf")]).sum().sum())

    print("\nSayisal sutunlarin ozeti:")
    print(numeric_data.describe().T)


if __name__ == "__main__":
    check_data_quality(Path("data/raw/jm1.csv"))