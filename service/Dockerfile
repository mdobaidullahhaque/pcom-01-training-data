# PCOM prediction service (Homework 6). Build from the REPO ROOT, not from service/:
#   docker build -f service/Dockerfile -t ml-prediction-api .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM's Linux wheel links libgomp.so.1 (OpenMP) but does not bundle it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY service/requirements.txt service/requirements.txt
RUN pip install -r service/requirements.txt

# Only what prediction needs (the same runtime files as tests/test_predict.py).
COPY config.py predict.py ./
COPY src/__init__.py src/
COPY src/model/__init__.py src/model/features.py src/model/counts.py src/model/
COPY models/pcom_entry_1/model.txt models/pcom_entry_1/model_meta.json models/pcom_entry_1/
COPY service/main.py service/

# Fail the build if the model cannot be loaded and scored (wrong hash, missing libgomp1, ...).
RUN python -m service.main --self-check

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
