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
- **XGBoost, Lstm, Prophet**: Tahmin modellerii.
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
    └── test_api.py      # API test dosyaları  '''
