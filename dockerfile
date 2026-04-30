FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Utiliser le miroir de Tsinghua (plus rapide depuis la Chine) ou le cache pip
RUN pip install --no-cache-dir --default-timeout=300 \
    --index-url https://pypi.org/simple/ \
    --retries=5 \
    -r requirements.txt

COPY app/ .

RUN mkdir -p /app/resultats /app/images /app/output

ENTRYPOINT ["python", "watermark_qim.py"]
CMD ["--demo"]