# PCOM prediction service (Homework 6). Build from the REPO ROOT, not from service/:
#   docker build -f service/Dockerfile -t ml-prediction-api .
#
# Full python:3.12 image, not python:3.12-slim: LightGBM's Linux wheel links libgomp.so.1 (OpenMP)
# without bundling it, and slim would need apt-get to install libgomp1. The FAU network blocks
# apt's HTTP (port 80) to deb.debian.org, so apt-get cannot run there. The full image already
# contains libgomp1 (a dependency of its gcc). pip uses HTTPS and works.
FROM python:3.12

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Fail early and clearly if a future base image drops libgomp.so.1.
RUN ldconfig -p | grep -q "libgomp.so.1" \
    || { echo "ERROR: libgomp.so.1 not found in the base image; LightGBM cannot load without it" >&2; exit 1; }

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
