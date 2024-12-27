from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from prophet import Prophet
from xgboost import XGBRegressor
import os
import optuna
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

# PostgreSQL bağlantı bilgileri
host = os.getenv("POSTGRES_HOST", "postgres")
database = os.getenv("POSTGRES_DB", "airflow")
user = os.getenv("POSTGRES_USER", "airflow")
password = os.getenv("POSTGRES_PASSWORD", "airflow")
port = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)

# Fourier özellikleri ekleme fonksiyonu
def add_fourier_features(df, column, period, n_harmonics=10):
    t = 2 * np.pi * df[column] / period
    for i in range(1, n_harmonics + 1):
        df[f'{column}_sin_{i}'] = np.sin(i * t)
        df[f'{column}_cos_{i}'] = np.cos(i * t)

# Veri yükleme işlevi
def load_energy_data(**context):
    # CSV dosyasını oku
    df = pd.read_csv('/opt/airflow/data/energydata_complete.csv')
    print("Yüklenen sütunlar:", df.columns.tolist())
    
    # CSV'deki sütunları kodda beklenen isimlere dönüştür
    df.rename(columns={
        'date': 'timestamp',
        'Appliances': 'appliances',
        'T1': 't1',
        'RH_1': 'rh_1',
        'T2': 't2',
        'RH_2': 'rh_2',
        'T3': 't3',
        'RH_3': 'rh_3',
        'T4': 't4',
        'RH_4': 'rh_4',
        'T5': 't5',
        'RH_5': 'rh_5',
        'T6': 't6',
        'RH_6': 'rh_6',
        'T7': 't7',
        'RH_7': 'rh_7',
        'T8': 't8',
        'RH_8': 'rh_8',
        'T9': 't9',
        'RH_9': 'rh_9',
        'T_out': 't_out',
        'Press_mm_hg': 'press_mm_hg',
        'RH_out': 'rh_out',
        'Windspeed': 'windspeed',
        'Visibility': 'visibility',
        'Tdewpoint': 'tdewpoint'
        # 'lights' zaten aynı isimde olduğundan rename gerekmiyor
        # 'rv1' ve 'rv2' de aynı
    }, inplace=True)
    
    # 'id' sütunu CSV'de yoksa, df'e ekleyebiliriz:
    df.insert(0, 'id', range(1, len(df) + 1))

    # Şimdi required_columns kontrole hazır
    required_columns = [
        'id', 'timestamp', 'appliances', 'lights', 't1', 'rh_1',
        't2', 'rh_2', 't3', 'rh_3', 't4', 'rh_4', 't5', 'rh_5',
        't6', 'rh_6', 't7', 'rh_7', 't8', 'rh_8', 't9', 'rh_9',
        't_out', 'press_mm_hg', 'rh_out', 'windspeed', 'visibility',
        'tdewpoint', 'rv1', 'rv2'
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    # Gerekli sütunları seç
    df = df[required_columns]
    
    # PostgreSQL'e yükle
    df.to_sql('energy_usage_cleaned', engine, if_exists='replace', index=False)
    print("Veri başarılı bir şekilde yüklendi.")


# Feature engineering işlevi
def veri_hazirla_features(**context):
    # Veriyi sorgudan çek
    query = "SELECT * FROM energy_usage_cleaned ORDER BY timestamp"
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])

    # DataFrame sütunlarını kontrol etmek için ekleyin
    print("Orijinal sütunlar:", df.columns.tolist())

    # Sütun türlerini doğru şekilde ayarlayın
    numeric_columns = [
        'appliances', 'lights', 't1', 'rh_1', 't2', 'rh_2', 't3', 'rh_3',
        't4', 'rh_4', 't5', 'rh_5', 't6', 'rh_6', 't7', 'rh_7',
        't8', 'rh_8', 't9', 'rh_9', 't_out', 'press_mm_hg',
        'rh_out', 'windspeed', 'visibility', 'tdewpoint', 'rv1',
        'rv2'
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eksik verileri kaldırın veya doldurun
    df.dropna(inplace=True)

    # Zaman temelli özellikler
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['season'] = ((df['month'] % 12 + 3) // 3)

    # Fourier özellikleri
    add_fourier_features(df, 'hour', 24)
    add_fourier_features(df, 'day_of_week', 7)

    # Gecikmeli özellikler
    df['appliances_lag1'] = df['appliances'].shift(1)
    df['appliances_lag6'] = df['appliances'].shift(6)
    df['appliances_lag12'] = df['appliances'].shift(12)

    # Hareketli ortalamalar
    df['appliances_rolling_mean_144'] = df['appliances'].rolling(window=144, min_periods=1).mean()

    # Anormallik Temizleme
    df['appliances_zscore'] = (df['appliances'] - df['appliances'].mean()) / df['appliances'].std()
    df = df[df['appliances_zscore'].abs() < 3]

    # DataFrame sütunlarını kontrol etmek için ekleyin
    print("İşlem sonrası sütunlar:", df.columns.tolist())

    # Veriyi yeni bir tabloya kaydetme
    df.to_sql('energy_usage_cleaned_features', engine, if_exists='replace', index=False)

    print("Feature engineering tamamlandı ve veri temizlendi.")

# XGBoost modelleme ve hiperparametre optimizasyonu
def optimize_xgboost(train, test, features, target):
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    def objective(trial):
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        }
        xgb = XGBRegressor(**param_grid)
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print("En iyi hiperparametreler:", best_params)

    # En iyi parametrelerle modeli eğit
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"XGBoost Optimum: MAE={mae}, RMSE={rmse}")
    return best_model, mae, rmse

def xgboost_modelleme(**context):
    train = pd.read_csv('/opt/airflow/data/train_features.csv')
    test = pd.read_csv('/opt/airflow/data/test_features.csv')
    target = 'appliances'
    features = [col for col in train.columns if col not in ['timestamp', 'appliances', 'id']]

    # Optimizasyon ve en iyi modelin alınması
    best_model, mae, rmse = optimize_xgboost(train, test, features, target)

    # Scaler'ı eğitme
    scaler = StandardScaler()
    scaler.fit(train[features])

    # Scaler'ı kaydetme
    with open('/opt/airflow/data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Modeli kaydetme
    with open('/opt/airflow/data/best_xgboost_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Eğitim verisi üzerindeki performansı kontrol et
    X_train, y_train = train[features], train[target]
    pred_train = best_model.predict(X_train)
    mae_train = mean_absolute_error(y_train, pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))

    print(f"Eğitim Verisi: MAE={mae_train}, RMSE={rmse_train}")

    # Cross-validation performansı
    cv_scores = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
    print(f"Cross-Validation MAE: {np.mean(-cv_scores)}")

    # XCom'a metrikleri gönder
    context['ti'].xcom_push(key='xgboost_metrics', value={'mae': mae, 'rmse': rmse})
    context['ti'].xcom_push(key='xgboost_train_metrics', value={'mae_train': mae_train, 'rmse_train': rmse_train})
    context['ti'].xcom_push(key='xgboost_cv_metrics', value={'cv_mae': np.mean(-cv_scores)})

# LSTM modelleme
def lstm_modelleme(**context):
    train = pd.read_csv('/opt/airflow/data/train_features.csv')
    test = pd.read_csv('/opt/airflow/data/test_features.csv')
    target = 'appliances'
    features = [col for col in train.columns if col not in ['timestamp', 'appliances', 'id']]

    def create_lstm_dataset(df, features, target, look_back=24):
        values = df[features + [target]].values
        X, Y = [], []
        for i in range(len(values) - look_back):
            X.append(values[i:i + look_back, :-1])
            Y.append(values[i + look_back, -1])
        return np.array(X), np.array(Y)

    look_back = 24
    X_train, y_train = create_lstm_dataset(train, features, target, look_back)
    X_test, y_test = create_lstm_dataset(test, features, target, look_back)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(look_back, len(features))),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[es, lr_reduction])

    pred = model.predict(X_test).flatten()
    pred = np.expm1(pred)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    print(f"LSTM: MAE={mae}, RMSE={rmse}")
    context['ti'].xcom_push(key='lstm_metrics', value={'mae': mae, 'rmse': rmse})
    model.save('/opt/airflow/data/optimized_lstm_model.h5')

# DAG Tanımlama
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'veri_isleme_dag',
    default_args=default_args,
    description='Prophet, XGBoost ve LSTM Model Pipeline',
    schedule_interval=timedelta(days=1),
)

# Görevler
load_energy_data_task = PythonOperator(
    task_id='load_energy_data',
    python_callable=load_energy_data,
    dag=dag,
)

veri_hazirla_features_task = PythonOperator(
    task_id='veri_hazirla_features',
    python_callable=veri_hazirla_features,
    dag=dag,
)

xgboost_modelleme_task = PythonOperator(
    task_id='xgboost_modelleme',
    python_callable=xgboost_modelleme,
    dag=dag,
)

lstm_modelleme_task = PythonOperator(
    task_id='lstm_modelleme',
    python_callable=lstm_modelleme,
    dag=dag,
)

# Bağımlılıklar
load_energy_data_task >> veri_hazirla_features_task >> [xgboost_modelleme_task, lstm_modelleme_task]
