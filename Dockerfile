FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py main.py

# Data klasörünü kopyalamıyoruz, çünkü volume ile mount ediyoruz.
# Model ve scaler docker-compose.yml ile volume olarak eklendi.
# Bu sayede container çalışırken /app/data içinde dosyalar hazır olacak.

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
