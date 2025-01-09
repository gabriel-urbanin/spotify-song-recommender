FROM --platform=linux/amd64 python:3.9-slim-bullseye as build

WORKDIR /app

COPY app/ /app/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model

CMD ["python", "/app/main.py"]