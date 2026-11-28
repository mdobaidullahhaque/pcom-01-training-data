"""PCOM prediction web service (Homework 6), based on riehlegroup/pcom-course-setup.

  GET  /health   -> {"health": "up"}
  POST /predict  {"testdata_download_endpoint": ..., "prediction_upload_endpoint": ...}
                 -> {"status": "prediction started"}; a background task downloads the input CSV,
                    runs run_model(df) and uploads prediction.csv as multipart field "predictions".

The prediction itself is predict.score_rows() (scorer D; float strategy EV, HR or HRMAX from
model_meta.json, default HRMAX since Homework 7; env PCOM_FLOAT_STRATEGY overrides it); nothing
is duplicated here. Run from the repo root:
  local:  .venv\\Scripts\\python.exe -m uvicorn service.main:app --port 8000
  Docker: see service/Dockerfile and docs/vm_deploy.md
  check:  python -m service.main --self-check   (exit 0 = model loads and scores)
"""
import io
import math
import os
import sys
import time
from pathlib import Path

# pandas must be imported before predict.load_lightgbm() runs: that function marks pandas as
# unavailable (to make predict.py start fast) if it has not been imported yet.
import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predict  # noqa: E402  (needs ROOT on sys.path)
from src.model import features as ft  # noqa: E402

TARGET_COLUMNS = list(predict.TARGET_COLUMNS)
KNOWN_COLUMNS = {name.lower(): name for name in (*ft.INPUT_COLUMNS, "purl", "prediction_date", *TARGET_COLUMNS)}
DECIMALS = 6
HTTP_TIMEOUT = 30
HTTP_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds; the wait grows with each attempt
FLOAT_STRATEGY_ENV = "PCOM_FLOAT_STRATEGY"

app = FastAPI(title="ML Prediction API")


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


def log(message):
    print(message, flush=True)


def frame_to_rows(df):
    """DataFrame -> list of dicts with schema column names and string values (None = missing).

    Headers are trimmed and matched case-insensitively to the schema spelling, as in predict.py.
    """
    names = [KNOWN_COLUMNS.get(str(name).strip().lower(), str(name).strip()) for name in df.columns]
    rows = []
    for values in df.itertuples(index=False, name=None):
        row = {}
        for name, value in zip(names, values):
            if name in row:
                continue  # a duplicated column: keep the first, as predict.read_rows does
            if value is None or (isinstance(value, float) and math.isnan(value)):
                row[name] = None
            else:
                text = str(value).strip()
                row[name] = None if text.lower() in predict.MISSING_TOKENS else text
        if any(value is not None for value in row.values()):
            rows.append(row)
    return rows


def choose_row(rows):
    """Index of the newest row by published_at; ties go to the later row; none readable -> last row."""
    best, best_time = len(rows) - 1, None
    for index, row in enumerate(rows):
        moment = ft.timestamp(row.get("published_at"))
        if moment is not None and (best_time is None or moment >= best_time):
            best, best_time = index, moment
    return best, best_time is not None


def clean(values):
    """Finite, non-negative floats rounded to DECIMALS; anything else becomes 0.0."""
    result = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        result.append(round(value, DECIMALS) + 0.0 if math.isfinite(value) and value > 0 else 0.0)
    return result


def float_strategy_override():
    """PCOM_FLOAT_STRATEGY (EV, HR or HRMAX) overrides model_meta.json, e.g. for a quick rollback to EV.

    Empty or unset: use default_float_strategy from model_meta.json. Unknown names are ignored.
    """
    value = os.environ.get(FLOAT_STRATEGY_ENV, "").strip().upper()
    if value and value not in predict.FLOAT_STRATEGIES:
        log(f"[run_model] ignoring {FLOAT_STRATEGY_ENV}={value!r} (not one of {', '.join(predict.FLOAT_STRATEGIES)})")
        return None
    return value or None


def predict_one(df):
    """Return (5 floats, info dict). Raises if the model cannot give a real prediction."""
    rows = frame_to_rows(df)
    if not rows:
        raise ValueError("the input has no data rows")
    index, by_date = choose_row(rows)
    row = rows[index]
    details = {}
    _, scores, failed, expected = predict.score_rows([row], None, float_strategy_override(), details)
    if failed or expected is None:
        raise RuntimeError("the model could not score the row" if failed else "model_meta.json has no EV settings")
    info = {
        "strategy": details.get("float_strategy", "?"),
        "rows": len(rows),
        "chosen": index + 1,
        "rule": "newest published_at" if by_date else "last row (no readable published_at)",
        "published_at": row.get("published_at"),
        "purl": row.get("purl"),
        "score": scores[0],
    }
    return clean(expected[0]), info


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    """One package's rows -> exactly one row with the 5 target columns. Never raises."""
    started = time.perf_counter()
    try:
        values, info = predict_one(df)
        log(f"[run_model] strategy={info['strategy']} rows={info['rows']} chosen=row {info['chosen']} ({info['rule']}, "
            f"published_at={info['published_at']}) score={info['score']:.4f} "
            f"output={values} time={time.perf_counter() - started:.2f}s")
    except Exception as error:
        values = [0.0] * len(TARGET_COLUMNS)
        log(f"[run_model] FALLBACK all zeros ({type(error).__name__}: {error}) "
            f"output={values} time={time.perf_counter() - started:.2f}s")
    return pd.DataFrame([values], columns=TARGET_COLUMNS)


def to_csv_text(prediction_df):
    """Plain US decimals: %.6f never writes exponents such as 2e-06."""
    buffer = io.StringIO()
    prediction_df.to_csv(buffer, sep=",", decimal=".", header=True, index=False,
                         float_format=f"%.{DECIMALS}f", lineterminator="\n")
    return buffer.getvalue()


def read_input(data):
    """CSV bytes -> DataFrame of strings. UTF-8 (with or without BOM), else Windows-1252."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        log("[predict] input is not valid UTF-8; reading it as Windows-1252")
        text = data.decode("cp1252", errors="replace")
    first_line = text.split("\n", 1)[0]
    separator = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=separator, decimal=".", dtype=str,
                       keep_default_na=False, skip_blank_lines=True)


def with_retries(action, what):
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            return action()
        except Exception as error:
            log(f"[predict] {what} failed (attempt {attempt}/{HTTP_ATTEMPTS}): {type(error).__name__}: {error}")
            if attempt == HTTP_ATTEMPTS:
                raise
            time.sleep(RETRY_DELAY * attempt)


def start_prediction(download_url: str, upload_url: str):
    """Download, predict, upload. A bad download or input still uploads a row of zeros."""
    started = time.perf_counter()
    log(f"[predict] started: download={download_url} upload={upload_url}")
    try:
        def download():
            response = requests.get(download_url, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            return response.content

        data = with_retries(download, "download")
        log(f"[predict] downloaded {len(data)} bytes in {time.perf_counter() - started:.2f}s")
        input_data = read_input(data)
    except Exception as error:
        log(f"[predict] no usable input ({type(error).__name__}: {error}); predicting zeros")
        input_data = pd.DataFrame()

    prediction_data = run_model(input_data)
    if len(prediction_data) != 1 or list(prediction_data.columns) != TARGET_COLUMNS:
        log("[predict] run_model returned the wrong shape; uploading zeros instead")
        prediction_data = pd.DataFrame([[0.0] * len(TARGET_COLUMNS)], columns=TARGET_COLUMNS)
    body = to_csv_text(prediction_data)

    try:
        def upload():
            files = {"predictions": ("prediction.csv", body, "text/csv")}
            response = requests.post(upload_url, files=files, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            return response

        response = with_retries(upload, "upload")
        try:
            result = response.json()
        except ValueError:
            result = response.text[:500]
        log(f"[predict] uploaded (HTTP {response.status_code}) in {time.perf_counter() - started:.2f}s total. "
            f"Evaluation result: {result}")
    except Exception as error:
        log(f"[predict] ERROR: upload gave up ({type(error).__name__}: {error})")


@app.post("/predict", status_code=200)
async def prediction(payload: EvaluationBody, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_prediction, payload.testdata_download_endpoint,
                              payload.prediction_upload_endpoint)
    return {"status": "prediction started"}


@app.get("/health", status_code=200)
async def health():
    return {"health": "up"}


def self_check():
    """Score one synthetic package with 10 past moderate GHSAs; it must reach the top EV bucket."""
    sample = pd.DataFrame([{"published_at": "2026-06-01T00:00:00.000Z", "ghsa_low_count": "0",
                            "ghsa_moderate_count": "10", "ghsa_high_count": "0", "ghsa_critical_count": "0"}])
    values, info = predict_one(sample)
    if not any(values):
        raise RuntimeError(f"self-check gave all zeros: {values}")
    log(f"[self-check] OK: strategy={info['strategy']} score={info['score']:.4f} output={values}")


if __name__ == "__main__":
    if "--self-check" in sys.argv[1:]:
        try:
            self_check()
        except Exception as error:
            log(f"[self-check] FAILED: {type(error).__name__}: {error}")
            sys.exit(1)
        sys.exit(0)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
