import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# PostgreSQL bağlantı bilgileri
host = os.getenv("POSTGRES_HOST", "postgres")
database = os.getenv("POSTGRES_DB", "airflow")
user = os.getenv("POSTGRES_USER", "airflow")
password = os.getenv("POSTGRES_PASSWORD", "airflow")
port = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

MODEL_PATH = "/app/data/best_xgboost_model.pkl"
SCALER_PATH = "/app/data/scaler.pkl"


# --- SQLAlchemy Model ---
class EnergyConsumptionCleanedFeatures(Base):
    __tablename__ = "energy_usage_cleaned_features"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    appliances = Column(Float)
    lights = Column(Float)
    t1 = Column(Float)
    rh_1 = Column(Float)
    t2 = Column(Float)
    rh_2 = Column(Float)
    t3 = Column(Float)
    rh_3 = Column(Float)
    t4 = Column(Float)
    rh_4 = Column(Float)
    t5 = Column(Float)
    rh_5 = Column(Float)
    t6 = Column(Float)
    rh_6 = Column(Float)
    t7 = Column(Float)
    rh_7 = Column(Float)
    t8 = Column(Float)
    rh_8 = Column(Float)
    t9 = Column(Float)
    rh_9 = Column(Float)
    t_out = Column(Float)
    press_mm_hg = Column(Float)
    rh_out = Column(Float)
    windspeed = Column(Float)
    visibility = Column(Float)
    tdewpoint = Column(Float)
    rv1 = Column(Float)
    rv2 = Column(Float)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Integer)
    month = Column(Integer)
    season = Column(Integer)
    hour_sin_1 = Column(Float)
    hour_cos_1 = Column(Float)

    # Gecikmeli (lag) sütunlarda NULL değer olabilmesi için nullable=True
    appliances_lag1 = Column(Float, nullable=True)
    appliances_lag6 = Column(Float, nullable=True)
    appliances_lag12 = Column(Float, nullable=True)
    appliances_rolling_mean_144 = Column(Float, nullable=True)
    # Diğer sütunlar...


# --- Pydantic (FastAPI) Model ---
class EnergyConsumptionResponse(BaseModel):
    id: int
    timestamp: datetime
    appliances: float
    lights: float
    t1: float
    rh_1: float
    t2: float
    rh_2: float
    t3: float
    rh_3: float
    t4: float
    rh_4: float
    t5: float
    rh_5: float
    t6: float
    rh_6: float
    t7: float
    rh_7: float
    t8: float
    rh_8: float
    t9: float
    rh_9: float
    t_out: float
    press_mm_hg: float
    rh_out: float
    windspeed: float
    visibility: float
    tdewpoint: float
    rv1: float
    rv2: float
    hour: int
    day_of_week: int
    is_weekend: int
    month: int
    season: int
    hour_sin_1: float
    hour_cos_1: float

    # Bu sütunlar veritabanında NULL dönebiliyor, o yüzden Optional.
    appliances_lag1: Optional[float] = None
    appliances_lag6: Optional[float] = None
    appliances_lag12: Optional[float] = None
    appliances_rolling_mean_144: Optional[float] = None

    class Config:
        orm_mode = True
        # Pydantic v2'de 'orm_mode' yerine 'from_attributes = True' kullanılabilir.


# --- Diğer Pydantic Modelleri ---
class PredictionInput(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float

class SavingsResponse(BaseModel):
    suggestions: List[str]

def load_xgboost_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("XGBoost Model ve Scaler başarıyla yüklendi.")
            return model, scaler
        except Exception as e:
            print(f"Model veya Scaler yüklenirken hata oluştu: {e}")
            return None, None
    else:
        print("Model veya Scaler dosyası bulunamadı.")
        return None, None

# Initialize model and scaler to None; they will be loaded on startup
model = None
scaler = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="Akıllı Ev Enerji Yönetim Sistemi API",
    description="Akıllı ev enerji yönetimi projesi için API."
)

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    print("Tablolar kontrol edildi ve gerekirse oluşturuldu.")
    # Model ve scaler'ı yeniden yüklemek için
    global model, scaler
    model, scaler = load_xgboost_model_and_scaler()

@app.get("/", summary="Ana Sayfa")
def read_root():
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")
    return {
        "message": "Model başarıyla yüklendi.",
        "en_iyi_model": "XGBOOST",
        "aciklama": "Bu API ile akıllı ev verilerinizden enerji tüketimi tahmini alabilir, tasarruf önerileri edinebilirsiniz.",
        "endpointler": {
            "/info": "Projenin amacı, yöntemleri ve tasarruf önerileri hakkında bilgi",
            "/data": "Temizlenmiş veriyi görüntülemek için",
            "/predict": "POST isteğiyle features göndererek tahmin almak için",
            "/reload-model": "Modeli ve scaler'ı yeniden yüklemek için",
            "/docs": "Swagger arayüzünü görüntülemek için"
        }
    }

@app.get("/data", response_model=List[EnergyConsumptionResponse], summary="Temizlenmiş Veriyi Getir")
def get_data(limit: int = 10, db: Session = Depends(get_db)):
    try:
        data = db.query(EnergyConsumptionCleanedFeatures).limit(limit).all()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veri çekilirken hata oluştu: {e}")

@app.post("/predict", response_model=PredictionResponse, summary="Tahmin Yap")
def predict(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="XGBoost Model veya Scaler yüklenmedi.")
    try:
        # Giriş verisini numpy array'e dönüştür
        X_new = np.array(data.features)
        
        # Giriş verisinin boyutunu kontrol et
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(1, -1)
        if X_new.shape[1] != 43:
            raise HTTPException(status_code=400, detail="Giriş verisi 43 özellik içermelidir.")
        
        # Giriş verisini scaler ile dönüştür
        X_new = scaler.transform(X_new)
        
        # Tahmin yap
        pred = model.predict(X_new)
        mean_prediction = pred.mean()  # Gerekirse ortalama alınabilir
        
        return {"prediction": float(mean_prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata oluştu: {e}")

@app.post("/reload-model", summary="Modeli Yeniden Yükle")
def reload_model_endpoint():
    global model, scaler
    model, scaler = load_xgboost_model_and_scaler()
    if model and scaler:
        return {"message": "Model ve Scaler başarıyla yeniden yüklendi."}
    else:
        raise HTTPException(status_code=500, detail="Model veya Scaler yeniden yüklenemedi.")

@app.get("/info", summary="Proje Bilgisi")
def get_info():
    return {
        "proje_amaci": (
            "Akıllı ev cihazlarından gelen enerji tüketim verilerini analiz ederek "
            "enerji tasarrufu önerileri sunmak."
        ),
        "yontem": (
            "Airflow: Otomatik veri pipeline. PostgreSQL: Veri saklama. "
            "XGBOOST modeli: Enerji tüketim tahmini. FastAPI: Tahmin ve veri erişimi için arayüz."
        ),
        "tasarruf_onerileri": (
            "Tahmin edilen tüketim değerlerine göre belirli saatlerde cihaz kullanımını azaltma, "
            "ısıtma/soğutma ayarlarını optimize etme."
        ),
        "iletisim": "admin@example.com"
    }

@app.post("/savings", response_model=SavingsResponse, summary="Tasarruf Önerileri Al")
def savings(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="XGBoost Model veya Scaler yüklenmedi.")
    try:
        # Giriş verisini numpy array'e dönüştür
        X_new = np.array(data.features)
        
        # Giriş verisinin boyutunu kontrol et
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(1, -1)
        
        # Giriş verisini scaler ile dönüştür
        X_new = scaler.transform(X_new)
        
        # Model tahmini
        predictions = model.predict(X_new).flatten()
        
        # Yüksek tüketim eşiklerini belirle
        threshold = 0.3
        high_consumption_periods = [f"{i}:00" for i, value in enumerate(predictions) if value > threshold]
        
        # Önerileri oluştur
        suggestions = [f"{period} saatlerinde cihaz kullanımını azaltın." for period in high_consumption_periods]
        
        # PostgreSQL'e önerileri kaydet
        with engine.connect() as connection:
            df_suggestions = pd.DataFrame({"hour": high_consumption_periods, "suggestion": suggestions})
            df_suggestions.to_sql("savings_suggestions", connection, if_exists="append", index=False)
        
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tasarruf önerileri hesaplanırken hata oluştu: {e}")
