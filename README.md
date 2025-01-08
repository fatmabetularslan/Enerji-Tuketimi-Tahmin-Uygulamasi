# Enerji Tüketimi Tahmin Uygulamas
Bu proje, enerji tüketimi tahminine yönelik bir **REST API** uygulamasıdır. **Python**, **FastAPI** ve **MinMaxScaler** gibi modern araçlar kullanılarak geliştirilmiştir. API, 24 zaman dilimi boyunca enerji tüketimi tahmin etmek için 43 özellikten oluşan giriş verilerini kabul eder ve çıktı olarak tahmin edilen değerleri döner.

## 🌐 Data ###

Bu proje, enerji tüketimi tahmini yapmak amacıyla **Appliances Energy Prediction** veri setini kullanmaktadır. Veriseti, bir evdeki enerji tüketimini ve çevresel faktörlerin enerji tüketimi üzerindeki etkilerini anlamak için oluşturulmuştur. 

**Veri Setinin Özellikleri:**
- **Veri Kaynağı**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Gözlemler**: 19,735 örnek veri
- **Zaman Dilimi**: Her 10 dakikada bir ölçülen enerji tüketimi değerleri
- **Özellikler**: Sıcaklık, nem, ışık seviyesi, rüzgar hızı gibi toplamda 29 çevresel değişken
- **Hedef Değişken**: Elektrikli ev aletlerinin enerji tüketimi (Appliances)

Veriseti, enerji verimliliğini artırmak ve enerji tüketimini optimize etmek amacıyla makine öğrenmesi modelleri için oldukça zengin bir veri kaynağıdır.

### Örnek Veri Kümesi

| Appliances | Temp_inside | Humidity | Light | Windspeed |
|------------|-------------|----------|-------|-----------|
| 50         | 21.5        | 40       | 200   | 1.5       |
| 60         | 22.0        | 42       | 210   | 2.0       |

> **Not**: Tabloda gösterilen değerler, veri setindeki örnek verilerdir.
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
AIRFLOW_PROJECT
├── airflow
│   ├── dags
│   ├── Dockerfile
│   └── requirements.txt
├── dags
├── data
│   └── energydata_complete.csv
├── fastapi
│   ├── Dockerfile
│   ├── main.py
└── docker-compose.yml

```
---

## 📊 Görselleştirme Analizleri

### 1. Özellik Önem Skoru ###

![image](https://github.com/user-attachments/assets/15a4e4f9-75cc-4852-b77f-d9676ee73601)

Analiz:

- Model, geçmiş enerji tüketim verilerini (%52.2) en önemli özellik olarak belirledi.*

- Appliances_lag1 ve Appliances_lag144, tahminler için kritik öneme sahip.*

---

### 2. Korelasyon Matrisi###

![image](https://github.com/user-attachments/assets/41164ab7-14ae-4dc4-93f0-802006d7b2d4)


Analiz:

- Sıcaklık ve nem özellikleri arasında yüksek korelasyon tespit edildi.
- Yüksek korelasyon gösteren değişkenler, çoklu doğrusal bağımlılığı önlemek için dikkatlice ele alınmıştır.

---

## #3. Enerji Tüketimi Dağılımı ### 

![image](https://github.com/user-attachments/assets/d1fc9380-526a-467a-b605-cecebb510143)

Analiz:
- Enerji tüketimi verisi büyük ölçüde 0-200 arasında yoğunlaşmıştır.
- Uç değerler temizlenmiş ve model performansı optimize edilmiştir.

---


## 🧑‍💻 API Kullanımı

### **Endpoint Açıklamaları**

| Endpoint                 | Metot | Açıklama                                               |
|--------------------------|-------|-------------------------------------------------------|
| `/predict`               | POST  | Enerji tüketimi tahmini yapmak için veriler gönderin. |
| `/data`                  | GET   | Veritabanından temizlenmiş veri alın.                 |
| `/energy-recommendation` | POST  | Tahmini enerji tüketimine dayalı öneriler alın.       |

---

### **cURL Örnekleri**
## 1.Veri Endpoint'i (/data)

Bu endpoint, veritabanında temizlenmiş enerji tüketimi verilerini almanızı sağlar.

```bash
curl -X GET "http://127.0.0.1:8000/data"
```
---
## 2. Tahmin Endpoint'i (`/predict`)

Bu endpoint, enerji tüketimini tahmin etmek için giriş verilerini kabul eder.

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
## Tahmin Çıktısı

![image](https://github.com/user-attachments/assets/30327be0-21ba-42f9-adb2-b0df7a021f7d)

*Tahmin edilen enerji tüketimi*

---

## 3. Öneri Endpoint'i (/energy-recommendation)

Bu endpoint, tahmini enerji tüketimi verilerine dayalı öneriler sağlar.
```bash
curl -X POST "http://127.0.0.1:8000/energy-recommendation" \
-H "Content-Type: application/json" \
-d '{
    "consumption": [100.0, 200.0, 300.0]
}'
```
---
## Öneri Çıktısı

![image](https://github.com/user-attachments/assets/d104d582-ee70-4492-a398-d37559e105bd)

*Enerji tasarrufu için öneriler*

---


## 🧑‍💻 Kurulum

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
## 🧪 Testler

Test dosyalarını çalıştırmak için:

```bash
pytest tests/

```
---



