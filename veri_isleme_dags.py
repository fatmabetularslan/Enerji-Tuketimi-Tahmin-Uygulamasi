from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from xgboost import XGBRegressor
import os

# PostgreSQL bağlantı bilgileri
host = os.getenv("POSTGRES_HOST", "postgres")
database = os.getenv("POSTGRES_DB", "airflow")
user = os.getenv("POSTGRES_USER", "airflow")
password = os.getenv("POSTGRES_PASSWORD", "airflow")
port = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)

def veri_hazirla(**context):
    df = pd.read_csv('/opt/airflow/data/energydata_complete.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df.columns = df.columns.str.lower()
    df.to_sql('energy_usage_prepared', engine, if_exists='replace', index=False)
    print("Veri hazırlama tamamlandı.")

def veri_temizle(**context):
    query = "SELECT * FROM energy_usage_prepared"
    df = pd.read_sql(query, engine)
    df.columns = df.columns.str.lower()
    if 'rv1' in df.columns:
        df.drop('rv1', axis=1, inplace=True)
    if 'rv2' in df.columns:
        df.drop('rv2', axis=1, inplace=True)

    df_prophet = df[['date','appliances']].rename(columns={'date':'ds','appliances':'y'})
    df_prophet.to_sql('energy_usage_prophet_ready', engine, if_exists='replace', index=False)
    print("Veri temizleme tamamlandı.")

def prophet_modelleme(**context):
    query = "SELECT ds, y FROM energy_usage_prophet_ready ORDER BY ds"
    df = pd.read_sql(query, engine, parse_dates=['ds'])
    forecast_steps = 48
    train = df.iloc[:-forecast_steps]
    test = df.iloc[-forecast_steps:]

    model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=False)
    model_prophet.fit(train)
    future = model_prophet.make_future_dataframe(periods=forecast_steps, freq='H')
    forecast = model_prophet.predict(future)

    pred = forecast.iloc[-forecast_steps:]['yhat'].values
    actual = test['y'].values
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"Prophet: MAE={mae}, RMSE={rmse}")

def veri_hazirla_features(**context):
    query = "SELECT * FROM energy_usage_prepared ORDER BY date"
    df = pd.read_sql(query, engine, parse_dates=['date'])

    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['month'] = df['date'].dt.month
    df['season'] = ((df['month']%12 + 3)//3)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    temp_cols = [c for c in df.columns if c.startswith('t') and c not in ['t_out','tdewpoint']]
    rh_cols = [c for c in df.columns if c.startswith('rh_') and c not in ['rh_out']]

    if len(temp_cols) > 0:
        df['t_avg_inside'] = df[temp_cols].mean(axis=1)
    else:
        df['t_avg_inside'] = np.nan

    if len(rh_cols) > 0:
        df['rh_avg_inside'] = df[rh_cols].mean(axis=1)
    else:
        df['rh_avg_inside'] = np.nan

    if 't_out' in df.columns:
        df['temp_diff'] = df['t_avg_inside'] - df['t_out']
    else:
        df['temp_diff'] = np.nan

    if 'lights' in df.columns:
        df['lights_binary'] = (df['lights'] > 0).astype(int)

    df['appliances_lag1'] = df['appliances'].shift(1)
    df['appliances_lag6'] = df['appliances'].shift(6)
    df['appliances_lag144'] = df['appliances'].shift(144)
    df['appliances_rolling_mean_144'] = df['appliances'].rolling(window=144, min_periods=1).mean()

    if 't_out' in df.columns:
        df['t_out_rolling_mean_144'] = df['t_out'].rolling(window=144, min_periods=1).mean()

    df = df.dropna()
    df['appliances'] = np.log1p(df['appliances'])

    df = df.reset_index(drop=True)
    df['id'] = df.index + 1

    # date -> timestamp
    df.rename(columns={'date':'timestamp'}, inplace=True)

    df.to_sql('energy_usage_cleaned', engine, if_exists='replace', index=False)
    print("Temizlenmiş veri 'energy_usage_cleaned' tablosuna yazıldı.")

    forecast_steps = 48
    train = df.iloc[:-forecast_steps]
    test = df.iloc[-forecast_steps:]

    features = [col for col in train.columns if col not in ['timestamp', 'appliances','id']]
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()

    train_scaled[features] = scaler.fit_transform(train[features])
    test_scaled[features] = scaler.transform(test[features])

    train_scaled.to_csv('/opt/airflow/data/train_features.csv', index=False)
    test_scaled.to_csv('/opt/airflow/data/test_features.csv', index=False)

    with open('/opt/airflow/data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Feature engineering tamamlandı, scaler kaydedildi ve train/test CSV'leri oluşturuldu.")

def xgboost_modelleme(**context):
    train = pd.read_csv('/opt/airflow/data/train_features.csv')
    test = pd.read_csv('/opt/airflow/data/test_features.csv')
    target = 'appliances'
    features = [col for col in train.columns if col not in ['timestamp','appliances','id']]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model_xgb = XGBRegressor(n_estimators=200, max_depth=5)
    model_xgb.fit(X_train, y_train)
    pred = model_xgb.predict(X_test)
    pred = np.expm1(pred)
    actual = np.expm1(y_test.values)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"XGBoost: MAE={mae}, RMSE={rmse}")

def lstm_modelleme(**context):
    train = pd.read_csv('/opt/airflow/data/train_features.csv')
    test = pd.read_csv('/opt/airflow/data/test_features.csv')
    target = 'appliances'
    features = [col for col in train.columns if col not in ['timestamp','appliances','id']]

    def create_lstm_dataset(df, features, target, look_back=24):
        values = df[features + [target]].values
        X, Y = [], []
        for i in range(len(values)-look_back):
            X.append(values[i:i+look_back, :-1])
            Y.append(values[i+look_back, -1])
        return np.array(X), np.array(Y)

    look_back = 24
    X_train, y_train = create_lstm_dataset(train, features, target, look_back)
    X_test, y_test = create_lstm_dataset(test, features, target, look_back)

    model_lstm = Sequential()
    model_lstm.add(LSTM(64, input_shape=(look_back, len(features))))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mse', optimizer='adam')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.1, callbacks=[es])

    pred = model_lstm.predict(X_test)
    pred = np.expm1(pred)
    actual = np.expm1(y_test)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"LSTM: MAE={mae}, RMSE={rmse}")

    model_lstm.save('/opt/airflow/data/best_lstm_model.h5')
    print("LSTM modeli kaydedildi.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

from airflow import DAG

dag = DAG(
    'veri_isleme_dag',
    default_args=default_args,
    description='Akıllı Ev Verileri - Prophet, XGBoost, LSTM Pipeline',
    schedule_interval=timedelta(days=1),
)

veri_hazirlama_task = PythonOperator(
    task_id='veri_hazirla',
    python_callable=veri_hazirla,
    dag=dag,
)

veri_temizleme_task = PythonOperator(
    task_id='veri_temizle',
    python_callable=veri_temizle,
    dag=dag,
)

veri_hazirla_features_task = PythonOperator(
    task_id='veri_hazirla_features',
    python_callable=veri_hazirla_features,
    dag=dag,
)

prophet_modelleme_task = PythonOperator(
    task_id='prophet_modelleme',
    python_callable=prophet_modelleme,
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

veri_hazirlama_task >> veri_temizleme_task >> veri_hazirla_features_task >> [xgboost_modelleme_task, lstm_modelleme_task] >> prophet_modelleme_task
