FROM python:3.7-slim

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_predict.txt requirements_predict.txt
COPY vast-flight-374515-36a0dca1ba5d.json vast-flight-374515-36a0dca1ba5d.json
COPY src/models/fastapi_predict.py fastapi_predict.py

WORKDIR /
RUN pip install -r requirements_predict.txt --no-cache-dir

CMD exec uvicorn fastapi_predict:app --port $PORT --host 0.0.0.0 --workers 1