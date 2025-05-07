FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/cache
RUN mkdir -p /app/data/wealth_profiles
RUN mkdir -p /app/data/market_data
RUN mkdir -p /app/data/investment_models

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8060"]

