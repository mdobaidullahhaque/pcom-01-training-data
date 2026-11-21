"""Tests for the Homework 6 web service (service/main.py) with the fake competition server.

service.main is imported at module level on purpose: pytest imports test modules before running
any test, so pandas is imported before predict.load_lightgbm() could mark it unavailable.
"""
import csv
import io
import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service import main as service
from service.fake_compy import TARGET_COLUMNS, FakeCompy, check_prediction_csv

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "tests" / "data" / "round1_input.csv"
ROUND1_EXPECTED = [0.060606, 0.363636, 0.121212, 0.030303, 0.0]
ZEROS = [0.0] * 5


@pytest.fixture
def client():
    return TestClient(service.app)


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(service, "RETRY_DELAY", 0)


def run_round(client, input_bytes, tmp_path, path="/input.csv"):
    """Serve input_bytes, POST /predict, return (response JSON, the upload record).

    TestClient runs the background task before post() returns, so the upload is already there.
    """
    with FakeCompy({"/input.csv": input_bytes}, out_dir=tmp_path) as compy:
        base = f"http://127.0.0.1:{compy.port}"
        response = client.post("/predict", json={
            "testdata_download_endpoint": base + path,
            "prediction_upload_endpoint": base + "/upload",
        })
        assert compy.upload_event.wait(30), "no upload received"
    assert response.status_code == 200
    assert len(compy.uploads) == 1
    return response.json(), compy.uploads[0]


def values_of(upload):
    records = list(csv.reader(io.StringIO(upload["text"])))
    assert records[0] == TARGET_COLUMNS
    assert len(records) == 2, "exactly one data row"
    assert all(value.strip() for value in records[1]), "no empty cells"
    values = [float(value) for value in records[1]]
    assert all(math.isfinite(value) and value >= 0 for value in values), "no NaN, no negatives"
    return values


def round1_without(columns):
    frame = pd.read_csv(ROUND1, dtype=str, keep_default_na=False)
    return frame.drop(columns=columns).to_csv(index=False).encode("utf-8")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "up"}


def test_round1_full_flow(client, tmp_path, capsys):
    answer, upload = run_round(client, ROUND1.read_bytes(), tmp_path)
    assert answer == {"status": "prediction started"}
    assert upload["problems"] == []
    assert upload["filename"] == "prediction.csv"
    assert upload["content_type"] == "text/csv"
    assert values_of(upload) == ROUND1_EXPECTED
    assert (tmp_path / "prediction.csv").read_text(encoding="utf-8") == upload["text"]
    log = capsys.readouterr().out
    assert "[run_model] rows=18 chosen=row 18 (newest published_at, published_at=2026-06-17T10:33:28.065Z)" in log
    assert "Evaluation result: {'status': 'received', 'valid': True" in log


def test_broken_csv_still_uploads_zeros(client, tmp_path, capsys):
    broken = b'package_created_at,published_at\n"2014-05-21,unterminated\n1,2,3,4\n'
    _, upload = run_round(client, broken, tmp_path)
    assert upload["problems"] == []
    assert values_of(upload) == ZEROS
    assert "FALLBACK all zeros" in capsys.readouterr().out


def test_failed_download_still_uploads_zeros(client, tmp_path, capsys):
    _, upload = run_round(client, b"", tmp_path, path="/missing.csv")  # 404
    assert upload["problems"] == []
    assert values_of(upload) == ZEROS
    log = capsys.readouterr().out
    assert "download failed (attempt 3/3)" in log


def test_empty_and_binary_input_upload_zeros(client, tmp_path):
    for index, data in enumerate([b"", b"\x00\xff\xfe garbage \x81\x00"]):
        _, upload = run_round(client, data, tmp_path / str(index))
        assert values_of(upload) == ZEROS


def test_missing_columns_still_work(client, tmp_path, capsys):
    # Without published_at, unpacked_size and file_count the last row is used; the GHSA history
    # is still there, so the package stays in the top bucket.
    _, upload = run_round(client, round1_without(["published_at", "unpacked_size", "file_count"]), tmp_path)
    assert upload["problems"] == []
    assert values_of(upload) == ROUND1_EXPECTED
    assert "chosen=row 18 (last row (no readable published_at)" in capsys.readouterr().out


def test_missing_ghsa_columns_give_a_valid_low_prediction(client, tmp_path):
    ghsa = [name for name in pd.read_csv(ROUND1, nrows=0).columns if name.startswith("ghsa_")]
    _, upload = run_round(client, round1_without(ghsa), tmp_path)
    assert upload["problems"] == []
    values = values_of(upload)
    assert values != ROUND1_EXPECTED and max(values) < 0.05


def test_run_model_picks_the_newest_row_in_any_order():
    frame = pd.read_csv(ROUND1, dtype=str, keep_default_na=False)
    newest = frame["published_at"].idxmax()
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    rows = service.frame_to_rows(shuffled)
    index, by_date = service.choose_row(rows)
    assert by_date and rows[index]["published_at"] == frame.loc[newest, "published_at"]
    result = service.run_model(shuffled)
    assert list(result.columns) == TARGET_COLUMNS and len(result) == 1
    assert result.iloc[0].tolist() == ROUND1_EXPECTED


def test_run_model_accepts_numeric_dataframes_like_the_template_reads_them():
    frame = pd.read_csv(ROUND1)  # pandas defaults: ints, floats and NaN instead of strings
    assert service.run_model(frame).iloc[0].tolist() == ROUND1_EXPECTED


def test_run_model_never_raises():
    for frame in [pd.DataFrame(), pd.DataFrame({"x": [None, None]}), pd.DataFrame({"published_at": ["junk"]})]:
        result = service.run_model(frame)
        assert list(result.columns) == TARGET_COLUMNS and len(result) == 1
        assert all(math.isfinite(value) and value >= 0 for value in result.iloc[0])


def test_clean_rounds_and_clamps():
    assert service.clean([0.06060606, -1, float("nan"), float("inf"), "x"]) == [0.060606, 0.0, 0.0, 0.0, 0.0]


def test_csv_text_never_uses_exponents():
    text = service.to_csv_text(pd.DataFrame([[0.000002, 1e-7, 0.0, 12.5, 0.0]], columns=TARGET_COLUMNS))
    assert text.splitlines()[1] == "0.000002,0.000000,0.000000,12.500000,0.000000"
    assert check_prediction_csv(text) == []


def test_header_matching_ignores_case_and_spaces():
    frame = pd.read_csv(ROUND1, dtype=str, keep_default_na=False)
    frame.columns = [f" {name.upper()} " for name in frame.columns]
    assert service.run_model(frame).iloc[0].tolist() == ROUND1_EXPECTED


def test_self_check_passes():
    service.self_check()


def test_fake_compy_checks_reject_bad_uploads():
    header = ",".join(TARGET_COLUMNS)
    assert check_prediction_csv(f"{header}\n0,1,0,0,0\n") == []
    assert check_prediction_csv(f"{header}\n0,1,0,0,0\n0,0,0,0,0\n")  # two rows
    assert check_prediction_csv(f"{header}\n0,nan,0,0,0\n")
    assert check_prediction_csv(f"{header}\n0,,0,0,0\n")
    assert check_prediction_csv("cve_high\n0.0\n")  # the template's example output


def test_docker_files_list_every_runtime_file():
    dockerfile = (ROOT / "service" / "Dockerfile").read_text(encoding="utf-8")
    ignore = (ROOT / "service" / "Dockerfile.dockerignore").read_text(encoding="utf-8")
    runtime = ["config.py", "predict.py", "src/__init__.py", "src/model/__init__.py", "src/model/features.py",
               "src/model/counts.py", "models/pcom_entry_1/model.txt", "models/pcom_entry_1/model_meta.json",
               "service/main.py", "service/requirements.txt"]
    for path in runtime:
        assert (ROOT / path).exists(), path
        assert f"!{path}" in ignore, path
        name = path.rsplit("/", 1)[-1]
        assert name in dockerfile, path
    instructions = [line.strip() for line in dockerfile.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert "FROM python:3.12" in instructions  # full image: it already contains libgomp1
    assert not any("apt-get" in line for line in instructions), "apt-get cannot reach Debian from the FAU network"
    assert any("ldconfig -p" in line and "libgomp.so.1" in line for line in instructions)
