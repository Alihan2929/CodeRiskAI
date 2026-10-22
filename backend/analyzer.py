from pathlib import Path

import lizard

SKIP_DIRS = {
    ".git", "build", "dist", "out", "target", ".venv", "venv",
    "node_modules", "__pycache__", "site-packages",
}

# lizard'in destekledigi ve arayuzde izin verilen kaynak kod uzantilari
SUPPORTED_EXTENSIONS = {
    ".java", ".py", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs", ".js",
    ".ts", ".go", ".php", ".rb", ".swift", ".kt", ".scala", ".rs", ".lua",
}

COMPLEXITY_LIMIT = 10
METHOD_LINES_LIMIT = 50
FILE_LINES_LIMIT = 300


def analyze_project(project_path: Path) -> dict:
    project_path = Path(project_path)
    if not project_path.exists() or not project_path.is_dir():
        raise FileNotFoundError(f"Proje klasoru bulunamadi: {project_path}")

    source_files = sorted(
        p for p in project_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part in SKIP_DIRS for part in p.parts)
    )

    files, methods = [], []
    for source_file in source_files:
        rel_file = str(source_file.relative_to(project_path))
        language = source_file.suffix.lower().lstrip(".")
        try:
            analysis = lizard.analyze_file(str(source_file))
            source = source_file.read_text(encoding="utf-8", errors="replace")
            non_empty_lines = sum(1 for line in source.splitlines() if line.strip())
            functions = list(analysis.function_list)
            max_complexity = max((f.cyclomatic_complexity for f in functions), default=0)
            long_methods = sum(1 for f in functions if f.nloc >= METHOD_LINES_LIMIT)

            warnings = []
            if non_empty_lines >= FILE_LINES_LIMIT:
                warnings.append("Uzun dosya")
            if max_complexity >= COMPLEXITY_LIMIT:
                warnings.append("Karmaşık metot")
            if long_methods:
                warnings.append("Uzun metot")

            files.append({
                "file": rel_file,
                "language": language,
                "non_empty_lines": non_empty_lines,
                "method_count": len(functions),
                "max_method_complexity": max_complexity,
                "long_method_count": long_methods,
                "long_file_warning": non_empty_lines >= FILE_LINES_LIMIT,
                "complex_method_warning": max_complexity >= COMPLEXITY_LIMIT,
                "warnings": warnings,
            })
            for fn in functions:
                methods.append({
                    "file": rel_file,
                    "language": language,
                    "method": fn.name,
                    "start_line": fn.start_line,
                    "end_line": fn.end_line,
                    "lines": fn.nloc,
                    "cyclomatic_complexity": fn.cyclomatic_complexity,
                    "complexity_warning": fn.cyclomatic_complexity >= COMPLEXITY_LIMIT,
                    "long_method_warning": fn.nloc >= METHOD_LINES_LIMIT,
                })
        except Exception as exc:  # tek bir bozuk dosya tum analizi durdurmasin
            files.append({
                "file": rel_file, "language": language, "error": str(exc),
                "non_empty_lines": 0, "method_count": 0,
                "max_method_complexity": 0, "long_method_count": 0,
                "long_file_warning": False, "complex_method_warning": False,
                "warnings": ["Dosya okunamadı"],
            })

    languages = {}
    for item in files:
        languages[item["language"]] = languages.get(item["language"], 0) + 1

    return {
        "files": files,
        "methods": methods,
        "summary": {
            "file_count": len(files),
            "method_count": sum(f.get("method_count", 0) for f in files),
            "long_file_count": sum(bool(f.get("long_file_warning")) for f in files),
            "complex_file_count": sum(bool(f.get("complex_method_warning")) for f in files),
            "complex_method_count": sum(bool(m.get("complexity_warning")) for m in methods),
            "long_method_count": sum(bool(m.get("long_method_warning")) for m in methods),
            "languages": languages,
        },
    }


# Eski ad ile uyumluluk
analyze_java_project = analyze_project
