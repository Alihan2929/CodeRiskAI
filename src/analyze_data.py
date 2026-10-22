from pathlib import Path
import pandas as pd


def analyze_defects(file_path):
    data = pd.read_csv(file_path)

    print("=== DEFECTS SINIF ANALIZI ===")

    counts = data["defects"].value_counts()

    print("\nSinif sayilari:")
    print(counts)

    print("\nSinif yuzdeleri:")
    print((data["defects"].value_counts(normalize=True) * 100).round(2))

    print("\nVeri tipi:")
    print(data["defects"].dtype)


if __name__ == "__main__":
    analyze_defects(Path("data/raw/jm1.csv"))