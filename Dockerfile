FROM python:3.9-slim-bullseye

WORKDIR /app

COPY app/ /app/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app/main.py"]