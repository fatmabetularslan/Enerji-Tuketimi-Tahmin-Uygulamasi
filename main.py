import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ------------------------------------------------------------------------------
# PostgreSQL bağlantı bilgileri
# ------------------------------------------------------------------------------
host = os.getenv("POSTGRES_HOST", "your_postgres_host")
database = os.getenv("POSTGRES_DB", "your_db")
user = os.getenv("POSTGRES_USER", "your_user")
password = os.getenv("POSTGRES_PASSWORD", "your_db")
port = os.getenv("POSTGRES_PORT", "your_port")
DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

MODEL_PATH = "/app/data/best_xgboost_model.pkl"
SCALER_PATH = "/app/data/scaler.pkl"


# ------------------------------------------------------------------------------
# SQLAlchemy Mode
# ------------------------------------------------------------------------------
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

    # Lag & Rolling
    appliances_lag1 = Column(Float, nullable=True)
    appliances_lag6 = Column(Float, nullable=True)
    appliances_lag12 = Column(Float, nullable=True)
    appliances_rolling_mean_144 = Column(Float, nullable=True)


# ------------------------------------------------------------------------------
# Pydantic (FastAPI) Models
# ------------------------------------------------------------------------------
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

    appliances_lag1: Optional[float] = None
    appliances_lag6: Optional[float] = None
    appliances_lag12: Optional[float] = None
    appliances_rolling_mean_144: Optional[float] = None

    class Config:
        orm_mode = True


# 43 özelliklik veriyi tahmin için alıyoruz
class PredictionInput(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: float



# feedback_text / did_apply opsiyonel
class RecommendationInput(BaseModel):
    features: List[float]
    feedback_text: Optional[str] = None
    did_apply: Optional[bool] = None


class SavingsResponse(BaseModel):
    suggestions: List[str]


# ------------------------------------------------------------------------------
# Model & Scaler Load
# ------------------------------------------------------------------------------
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


model = None
scaler = None


# ------------------------------------------------------------------------------
# DB session
# ------------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# FastAPI Uygulaması
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Appliances Energy Project - All in One",
    description="""Bu projede, /data ile veriyi çekebilir, /predict ile tahmin,
    /energyRecommendations ile tasarruf önerileri + feedback kaydı yapabilirsiniz."""
)


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    print("Tablolar kontrol edildi ve gerekirse oluşturuldu.")
    global model, scaler
    model, scaler = load_xgboost_model_and_scaler()


# ------------------------------------------------------------------------------
# Ana sayfa
# ------------------------------------------------------------------------------
@app.get("/")
def read_root():
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")
    return {
            "message": "Model başarıyla yüklendi. Gelişmiş sürüm",
            "endpointler": {
            "/info": "Proje detay bilgisi",
            "/data": "Veritabanından temizlenmiş veri",
            "/predict": "XGBoost tahmini",
            "/energyDocumentation": "Gelişmiş tasarruf önerileri",
    
            "/docs": "Swagger/OpenAPI UI"
    }}


# ------------------------------------------------------------------------------
# /data endpoint: veritabanındaki kayıtlardan ilk 'limit' kadarını getirir
# ------------------------------------------------------------------------------
@app.get("/data", response_model=List[EnergyConsumptionResponse])
def get_data(limit: int = 10, db: Session = Depends(get_db)):
    try:
        data = db.query(EnergyConsumptionCleanedFeatures).limit(limit).all()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# /predict endpoint: Tahmin
# ------------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model veya Scaler yüklenmedi.")
    try:
        X_arr = np.array(data.features)

        # Çok satırlı veri desteği
        if len(X_arr.shape) == 1:
            total_len = X_arr.shape[0]
            if total_len % 43 != 0:
                raise HTTPException(
                    status_code=400,
                    detail="Giriş verisi 43'ün katı uzunlukta olmalı (her satır 43)."
                )
            n_rows = total_len // 43
            X_arr = X_arr.reshape(n_rows, 43)
        else:
            if X_arr.shape[1] != 43:
                raise HTTPException(
                    status_code=400,
                    detail="Her satır 43 özellik içermeli."
                )

        X_scaled = scaler.transform(X_arr)
        preds = model.predict(X_scaled)
        mean_prediction = preds.mean()

        return {"prediction": float(mean_prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata: {e}")


# ------------------------------------------------------------------------------
# /energyRecommendations: Gelişmiş tasarruf önerileri + opsiyonel feedback
# ------------------------------------------------------------------------------
@app.post("/energyRecommendations", response_model=SavingsResponse)
def energy_recommendations(data: RecommendationInput, db: Session = Depends(get_db)):
    """
    1) Model tahmini + tasarruf önerileri
    2) Eğer 'feedback_text' veya 'did_apply' doluysa user_feedback tablosuna kaydet
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model veya Scaler yüklenmedi.")

    try:
        # --- (A) Tahmin/Öneri ---
        X_arr = np.array(data.features)

        if len(X_arr.shape) == 1:
            total_len = X_arr.shape[0]
            if total_len % 43 != 0:
                raise HTTPException(
                    status_code=400,
                    detail="Giriş verisi (features) 43'ün katı olmalı."
                )
            n_rows = total_len // 43
            X_arr = X_arr.reshape(n_rows, 43)
        else:
            if X_arr.shape[1] != 43:
                raise HTTPException(
                    status_code=400,
                    detail="Her satır 43 özellik içermeli."
                )

        X_scaled = scaler.transform(X_arr)
        predictions = model.predict(X_scaled).flatten()

        # Eşik değerleri
        threshold_high = 0.4
        threshold_medium = 0.2
        dynamic_mean = 0.3
        dynamic_std = 0.1
        threshold_dynamic = dynamic_mean + dynamic_std  # 0.4

        total_consumption = predictions.sum()
        target_savings_rate = 0.1  # %10
        optimal_consumption = total_consumption * (1 - target_savings_rate)
        diff = total_consumption - optimal_consumption
        estimated_bill = total_consumption  # 1 kWh = 1 TL varsayımı

        suggestions = []

        # her satır için
        for i, val in enumerate(predictions):
            hour_suggestions = []
            if val > threshold_high:
                hour_suggestions.append(f"{i}:00 saatinde tüketim yüksek.")
            elif val > threshold_medium:
                hour_suggestions.append(f"{i}:00 saatinde tüketim orta.")
            else:
                hour_suggestions.append(f"{i}:00 saatinde tüketim düşük.")

            if val > threshold_dynamic:
                hour_suggestions.append(f"{i}:00 saatinde dinamik eşik aşıldı.")

            # saat (i) sabah/akşam vs.
            if 7 <= i < 9:
                hour_suggestions.append("Sabah (07-09) tasarrufu.")
            elif 18 <= i < 22:
                hour_suggestions.append("Akşam (18-22) tasarrufu.")

            suggestions.append(" / ".join(hour_suggestions))

        # hedef odaklı
        if total_consumption > optimal_consumption:
            suggestions.append(f"Günlük hedef tüketiminizden {diff:.2f} kWh fazladasınız.")
        else:
            suggestions.append("Tebrikler! %10 tasarruf hedefini yakaladınız.")

        # fatura tahmini
        suggestions.append(f"Tahmini tüketim: {total_consumption:.2f} kWh, ~{estimated_bill:.2f} TL.")

        # ilk n satır önerisini tabloya yazma
        n_rows = len(predictions)
        store_suggestions = suggestions[:n_rows]
        with engine.connect() as con:
            df_suggestions = pd.DataFrame({
                "hour": [f"{i}:00" for i in range(n_rows)],
                "suggestion": store_suggestions
            })
            df_suggestions.to_sql("savings_suggestions", con=con, if_exists="append", index=False)

        # (B) Feedback Kaydetme (opsiyonel) 
        if data.feedback_text is not None or data.did_apply is not None:
            feedback_dict = {
                "suggestion_text": [data.feedback_text or ""],
                "did_apply": [data.did_apply if data.did_apply is not None else False],
                "timestamp": [datetime.utcnow()]
            }
            df_feedback = pd.DataFrame(feedback_dict)
            with engine.connect() as con:
                df_feedback.to_sql("user_feedback", con=con, if_exists="append", index=False)

        return {"suggestions": suggestions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tasarruf önerileri veya feedback sırasında hata: {e}")


# ------------------------------------------------------------------------------
# /reload-model
# ------------------------------------------------------------------------------
@app.post("/reload-model")
def reload_model_endpoint():
    global model, scaler
    model, scaler = load_xgboost_model_and_scaler()
    if model and scaler:
        return {"message": "Model ve Scaler yeniden yüklendi."}
    else:
        raise HTTPException(status_code=500, detail="Model veya Scaler yeniden yüklenemedi.")
