
FROM python:3.9-slim

EXPOSE $PORT


COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*





WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir

CMD uvicorn  src.production.prediction_api:app --port 8080 --host 0.0.0.0 --workers 1