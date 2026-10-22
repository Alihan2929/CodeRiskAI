from pathlib import Path
import pandas as pd


def inspect_dataset(file_path):
    path = Path(file_path)

    if not path.exists():
        print("Hata: Veri dosyasi bulunamadi!")
        return

    data = pd.read_csv(path)

    print("=== CODERISK AI ===")
    print(f"Satir sayisi: {data.shape[0]}")
    print(f"Sutun sayisi: {data.shape[1]}")

    print("\nSutun isimleri:")
    print(data.columns.tolist())

    print("\nIlk 5 kayit:")
    print(data.head())

    print("\nEksik degerler:")
    print(data.isnull().sum())


if __name__ == "__main__":
    inspect_dataset("data/raw/jm1.csv")