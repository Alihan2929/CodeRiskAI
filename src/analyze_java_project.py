
from pathlib import Path

import pandas as pd
import lizard


OUTPUT_DIR = Path("reports")


def analyze_java_project(project_path):
    project_path = Path(project_path)

    if not project_path.exists() or not project_path.is_dir():
        raise FileNotFoundError(
            f"Proje klasoru bulunamadi: {project_path}"
        )

    java_files = [
        path for path in project_path.rglob("*.java")
        if not any(
            folder in {".git", "build", "dist", "out", "target"}
            for folder in path.parts
        )
    ]

    if not java_files:
        print("Bu klasorde Java dosyasi bulunamadi.")
        return

    file_results = []
    method_results = []

    for java_file in java_files:
        try:
            analysis = lizard.analyze_file(str(java_file))
            source = java_file.read_text(
                encoding="utf-8", errors="replace"
            )
            lines = source.splitlines()

            file_loc = sum(
                1 for line in lines if line.strip()
            )

            file_complexity = max(
                (
                    function.cyclomatic_complexity
                    for function in analysis.function_list
                ),
                default=0,
            )

            file_results.append({
                "file": str(java_file),
                "non_empty_lines": file_loc,
                "method_count": len(analysis.function_list),
                "max_method_complexity": file_complexity,
                "long_file_warning": file_loc >= 300,
                "complex_method_warning": file_complexity >= 10,
            })

            for function in analysis.function_list:
                method_results.append({
                    "file": str(java_file),
                    "method": function.name,
                    "start_line": function.start_line,
                    "end_line": function.end_line,
                    "lines": function.nloc,
                    "cyclomatic_complexity":
                        function.cyclomatic_complexity,
                    "complexity_warning":
                        function.cyclomatic_complexity >= 10,
                    "long_method_warning": function.nloc >= 50,
                })

        except (OSError, UnicodeError) as error:
            print(f"Dosya okunamadi: {java_file}: {error}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files_df = pd.DataFrame(file_results)
    methods_df = pd.DataFrame(method_results)

    files_df.to_csv(
        OUTPUT_DIR / "java_file_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    methods_df.to_csv(
        OUTPUT_DIR / "java_method_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== CODERISK AI: JAVA PROJE ANALIZI ===")
    print("Taranan Java dosyasi:", len(files_df))

    if not files_df.empty:
        print("Toplam metot:", int(files_df["method_count"].sum()))
        print(
            "Uzun dosya uyarisi:",
            int(files_df["long_file_warning"].sum()),
        )
        print(
            "Karmasik metot iceren dosya:",
            int(files_df["complex_method_warning"].sum()),
        )

    if not methods_df.empty:
        print(
            "Karmasik metot sayisi:",
            int(methods_df["complexity_warning"].sum()),
        )
        print(
            "Uzun metot sayisi:",
            int(methods_df["long_method_warning"].sum()),
        )

        print("\n=== EN KARMASIK METOTLAR ===")
        print(
            methods_df.sort_values(
                "cyclomatic_complexity", ascending=False
            ).head(10).to_string(index=False)
        )

    print("\nRaporlar kaydedildi:")
    print("reports/java_file_analysis.csv")
    print("reports/java_method_analysis.csv")


if __name__ == "__main__":
    folder = input(
        "Analiz edilecek Java proje klasorunun tam yolunu gir: "
    ).strip().strip('"')

    analyze_java_project(folder)
