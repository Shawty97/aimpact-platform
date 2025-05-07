# Build stage for Python backend
FROM python:3.11-slim AS backend-build

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build stage for Next.js dashboard
FROM node:18-alpine AS dashboard-build

WORKDIR /app

# Copy dashboard files
COPY dashboard/package*.json ./
RUN npm ci

COPY dashboard/ .
RUN npm run build

# Production stage for Python backend
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from build stage
COPY --from=backend-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-build /usr/local/bin /usr/local/bin

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY backend/ /app/
COPY --from=dashboard-build /app/.next /app/static/dashboard

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HOST=0.0.0.0

# Create non-root user for security
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    pkg-config \
    gcc \
    g++ \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p /app/models/tts/coqui \
    && mkdir -p /app/models/stt/vosk

# Download Vosk model
RUN curl -LO https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip \
    && unzip vosk-model-en-us-0.22.zip -d /app/models/stt/vosk \
    && rm vosk-model-en-us-0.22.zip

# Install specialized billionaire-focused packages
RUN pip install --no-cache-dir \
    yahooquery \
    yfinance \
    finnhub-python \
    alpha_vantage \
    networkx \
    plotly \
    dash \
    pandas_datareader \
    pandas \
    sec-edgar-downloader \
    python-linkedin-v2

# Copy application code
COPY . .

# Create necessary directory structure for billionaire features
RUN mkdir -p /app/data/market_analysis \
    && mkdir -p /app/data/wealth_tracking \
    && mkdir -p /app/data/network_analysis \
    && mkdir -p /app/workflows/billionaire_templates

# Initialize the billionaire workflows
RUN echo '{"name": "Billionaire Wealth Tracking", "version": "1.0.0"}' > /app/workflows/billionaire_templates/wealth_tracking.json \
    && echo '{"name": "Network Expansion", "version": "1.0.0"}' > /app/workflows/billionaire_templates/network_expansion.json \
    && echo '{"name": "Market Opportunity Analysis", "version": "1.0.0"}' > /app/workflows/billionaire_templates/market_analysis.json \
    && echo '{"name": "Competitor Intelligence", "version": "1.0.0"}' > /app/workflows/billionaire_templates/competitor_intelligence.json \
    && echo '{"name": "Wealth Preservation Strategy", "version": "1.0.0"}' > /app/workflows/billionaire_templates/wealth_preservation.json

# Set the entrypoint
ENTRYPOINT ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]

