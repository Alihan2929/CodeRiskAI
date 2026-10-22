from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile, BadZipFile
import shutil

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .analyzer import SKIP_DIRS, SUPPORTED_EXTENSIONS, analyze_project
    from . import model_service
except ImportError:
    from analyzer import SKIP_DIRS, SUPPORTED_EXTENSIONS, analyze_project
    import model_service

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="CodeRiskAI API", version="0.2.0")
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024
MAX_ZIP_FILES = 10000


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "CodeRiskAI API"}


@app.get("/api/model-info")
def model_info():
    try:
        return model_service.get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/sample")
def sample():
    try:
        return model_service.get_sample()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/predict")
def predict(metrics: dict = Body(...)):
    try:
        return model_service.predict(metrics)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="En az bir kod dosyası veya .zip yükle.")
    with TemporaryDirectory(prefix="coderiskai-") as temp:
        root = Path(temp) / "project"
        root.mkdir()
        total = 0
        source_count = 0
        for index, upload in enumerate(files):
            name = Path(upload.filename or "upload").name
            suffix = Path(name).suffix.lower()
            content = await upload.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"Dosya çok büyük: {name} (25 MB sınırı).")
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Toplam yükleme 60 MB sınırını aşıyor.")
            if suffix in SUPPORTED_EXTENSIONS:
                folder = root / f"yuklenen_{index}"
                folder.mkdir(exist_ok=True)
                (folder / name).write_bytes(content)
                source_count += 1
            elif suffix == ".zip":
                archive = Path(temp) / f"upload_{index}.zip"
                archive.write_bytes(content)
                try:
                    with ZipFile(archive) as zf:
                        infos = [i for i in zf.infolist() if not i.is_dir()]
                        if len(infos) > MAX_ZIP_FILES:
                            raise HTTPException(status_code=413, detail="ZIP içinde çok fazla dosya var.")
                        for info in infos:
                            rel = PurePosixPath(info.filename)
                            if rel.is_absolute() or ".." in rel.parts:
                                continue
                            if rel.suffix.lower() not in SUPPORTED_EXTENSIONS:
                                continue
                            if any(part in SKIP_DIRS for part in rel.parts):
                                continue
                            if info.file_size > MAX_FILE_BYTES:
                                continue
                            dest = root.joinpath(f"zip_{index}", *rel.parts)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(info) as src, dest.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
                            source_count += 1
                except BadZipFile:
                    raise HTTPException(status_code=400, detail=f"Geçersiz ZIP dosyası: {name}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Desteklenmeyen dosya: {name}. Kod dosyası veya .zip yükle.",
                )
        if source_count == 0:
            raise HTTPException(status_code=400, detail="Yüklenen içerikte analiz edilebilir kod dosyası bulunamadı.")
        return analyze_project(root)


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
