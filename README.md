# CodeRiskAI

Yazılım kodundaki hata riskini analiz eden proje: JM1 verisiyle eğitilmiş Random Forest modeli + kod karmaşıklığı analizi + web arayüzü.

## Kurulum ve çalıştırma (Windows / VS Code)

Proje klasöründe (içinde `backend`, `frontend`, `src` klasörleri olan yer) terminal aç:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Sonra tarayıcıda şu adresi aç: http://127.0.0.1:8000

Sunucuyu durdurmak için terminalde `Ctrl + C`.

## Arayüz sayfaları
- **Genel Bakış:** Modelin gerçek test sonuçları (accuracy, precision, recall, ROC-AUC), özellik önemleri ve eşik karşılaştırması. Veriler `reports/` klasöründen okunur.
- **Kod Analizi:** Kod dosyaları veya `.zip` proje arşivi yükle. Dosya/metot sayısı, satır sayısı, cyclomatic complexity, uzun ve karmaşık metot uyarıları hesaplanır. Java, Python, C/C++, C#, JavaScript, TypeScript, Go, PHP, Ruby, Swift, Kotlin, Scala, Rust, Lua desteklenir (lizard).
- **Raporlar:** Henüz backend'e bağlı değil (analiz geçmişi saklanmıyor).
- **Model ve Tahmin:** 21 JM1 metriğini girerek ya da test verisinden rastgele örnek yükleyerek kusur olasılığını hesapla.

## API uçları
| Uç | Ne yapar |
|---|---|
| `GET /api/health` | Sunucu çalışıyor mu |
| `GET /api/model-info` | Model bilgisi ve rapor sonuçları |
| `GET /api/sample` | Test verisinden rastgele kayıt |
| `POST /api/predict` | 21 metriği JSON olarak al, risk tahmini döndür |
| `POST /api/analyze` | Kod dosyaları / ZIP al, statik analiz döndür |

Otomatik test sayfası: http://127.0.0.1:8000/docs

## Sınırlar (dürüst not)
- ML modeli **C kodundan gelen JM1 metrikleriyle** eğitildi. Yüklenen kod dosyaları için ML tahmini üretilmez; "Kod Analizi" kural tabanlı (statik) çalışır. Bu iki parçanın birleştirilmesi sonraki aşamadır.
- Tahmin sonucu doğrulanmış hata olasılığı değil, risk skorudur.
- Yükleme limitleri: tek dosya 25 MB, toplam 60 MB. Yerel geliştirme içindir, internete açık servis olarak yayımlama.
- `models/*.joblib` dosyaları eğitildiği scikit-learn sürümüyle uyumlu bir sürümle yüklenmelidir; uyarı çıkarsa `src/train_model.py` ile modeli yeniden eğit.
