# Enerji Tüketimi Tahmin Uygulaması

Bu proje, enerji tüketimi tahminine yönelik bir **REST API** uygulamasıdır. **Python**, **FastAPI** ve **MinMaxScaler** gibi modern araçlar kullanılarak geliştirilmiştir. API, 24 zaman dilimi boyunca enerji tüketimi tahmin etmek için 43 özellikten oluşan giriş verilerini kabul eder ve çıktı olarak tahmin edilen değerleri döner.

---

## 🚀 Proje Özellikleri

- **RESTful API**: FastAPI framework'ü ile hızlı ve güvenilir bir API.
- **Veri Ölçekleme**: MinMaxScaler ile verilerin normalize edilmesi.
- **Tahmin Modeli**: XGBoost gibi güçlü bir makine öğrenmesi modeli kullanılarak enerji tahminleri yapılır.
- **Docker Desteği**: Proje, kolay dağıtım ve yönetim için Docker ile paketlenmiştir.
- **Geliştirilebilir Altyapı**: API, ölçeklenebilir ve genişletilebilir bir yapı sunar.

---

## 🛠️ Kullanılan Teknolojiler

- **Python**: Ana geliştirme dili.
- **FastAPI**: API'nin geliştirilmesi için kullanılan framework.
- **XGBoost, Lstm, Prophet**: Tahmin modelleri.
- **NumPy / Pandas**: Veri işleme ve manipülasyon.
- **Scikit-learn**: Veri ölçekleme ve model değerlendirme.
- **Docker**: Uygulamanın taşınabilirliğini sağlamak için konteyner teknolojisi.
- **cURL**: API testleri için kullanılan araç.

---

## 📂 Proje Yapısı

```plaintext
.
├── app/
│   ├── main.py          # FastAPI uygulamasının ana dosyası
│   ├── model.pkl        # Eğitilmiş tahmin modeli
│   └── utils.py         # Yardımcı fonksiyonlar
├── requirements.txt     # Bağımlılıklar
├── Dockerfile           # Docker yapılandırması
├── README.md            # Proje dokümantasyonu
└── tests/
    └── test_api.py      # API test dosyaları
```
---

##🧑‍💻 Kurulum

**1. Depoyu Klonlayın**
```bash
git clone https://github.com/kullanici_adi/enerji-tahmin-api.git
cd enerji-tahmin-api
```

---
**2. Sanal Ortamı Kurun**
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
pip install -r requirements.txt
```
---
**3. API'yi Çalıştırın**
```bash
uvicorn app.main:app --reload
```
---

**4. API'yi Test Edin**
```bash

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "features": [
        [744.2, 75.0, 7.2, 29.0, 2.3, -0.9659, -0.2588, 17.3, 47.47, 10.8, 0.0, 40.0, 30.0, 60.0, 104.1, 7.0, 40.0, 30.0, 60.0, 104.1, 7.0, 40.0, 30.0, 60.0, 20.07, 42.83, 19.0, 42.41, 19.79, 44.7, 19.26, 42.56, 17.6, 50.9, 6.16, 76.89, 18.14, 37.91, 18.6, 45.79, 17.1]
    ]
}'

```
---
## 🐳 Docker Kullanımı

**1. Docker İmajını Oluşturun**
```bash
docker build -t enerji-tahmin-api .
```
---

**2. Docker Konteynerini Çalıştırın**
```bash
docker run -p 8000:8000 enerji-tahmin-api

```
---
## 📊 Örnek Tahmin Çıktısı

```bash
{
    "features": [
        [744.2, 75.0, 7.2, 29.0, 2.3, -0.9659, -0.2588, 17.3, 47.47, 10.8, 0.0, 40.0, 30.0, 60.0, 104.1, 7.0, 40.0, 30.0, 60.0, 104.1, 7.0, 40.0, 30.0, 60.0, 20.07, 42.83, 19.0, 42.41, 19.79, 44.7, 19.26, 42.56, 17.6, 50.9, 6.16, 76.89, 18.14, 37.91, 18.6, 45.79, 17.1]
    ]
}

```
---
**Tahmin Çıktısı**

```bash
{
    "predictions": [62.5]
}


```
---
## 🧪 Testler

Test dosyalarını çalıştırmak için:

```bash
pytest tests/

```
---



