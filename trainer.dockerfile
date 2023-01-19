# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data.dvc data.dvc

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://cats-and-dogs-dtumlops/
RUN dvc pull
RUN wandb login --relogin e77f8847024e733430e6d041cf05985cd459f692

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
