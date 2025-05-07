FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgraphviz-dev \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for entity recognition
RUN python -m spacy download en_core_web_lg

RUN mkdir -p /app/cache

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8070"]

