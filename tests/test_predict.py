"""Tests for predict.py. Each run is a separate process in a temporary copy of the runtime files,
which also checks that predict.py works from any folder and needs nothing else from the repo."""
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

import config
from src.model import counts as ct
from src.model import features as ft

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    "predict.py", "config.py", "src/__init__.py", "src/model/__init__.py", "src/model/features.py",
    "src/model/counts.py", "models/pcom_entry_1/model.txt", "models/pcom_entry_1/model_meta.json",
]
TARGETS = [f"target_ghsa_{severity}_count_90d" for severity in config.SEVERITIES]
ZERO = [0, 0, 0, 0, 0]


@pytest.fixture(scope="module")
def training_rows():
    with open(config.OUTPUT_CSV, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def core_columns(training_rows):
    return [name for name in training_rows[0] if name != "prediction_date"]


@pytest.fixture
def clone(tmp_path):
    """A copy of only the files predict.py needs at run time, as a fresh clone would have them."""
    target = tmp_path / "clone"
    for name in RUNTIME_FILES:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target / name)
    return target


def write_csv(path, header, rows, delimiter=",", encoding="utf-8", newline="\n"):
    with open(path, "w", newline="", encoding=encoding) as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator=newline)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def run(clone, input_path, *extra, cwd=None):
    """Run predict.py from a different folder; return (exit code, output rows, screen text)."""
    output = input_path.with_name("out.csv")
    result = subprocess.run(
        [sys.executable, str(clone / "predict.py"), str(input_path), "-o", str(output), *extra],
        capture_output=True, text=True, cwd=cwd or input_path.parent, timeout=60,
    )
    rows = []
    if output.exists():
        with open(output, newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
        assert table[0] == TARGETS
        rows = [[int(value) for value in row] for row in table[1:]]
    return result.returncode, rows, result.stdout + result.stderr


def expected_counts(rows):
    """Counts computed in this process with the training code path."""
    meta = json.loads((config.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    X, names = ft.build_features(rows, None, use_reference_features=False)
    booster = lgb.Booster(model_file=str(config.MODEL_DIR / "model.txt"))
    scores = np.nan_to_num(X[:, names.index("ghsa_total")]) + 0.5 * booster.predict(X)
    return ct.predict_counts(scores, meta["counts"]["strategies"], meta["counts"]["default_strategy"]).tolist()


def sample(training_rows, count, seed=0):
    rng = random.Random(seed)
    picked = [row for row in training_rows if row["purl"].startswith(("pkg:npm/electron@", "pkg:npm/next@"))]
    return picked + rng.sample(training_rows, count - len(picked))


# 1. Class-like input ------------------------------------------------------------------------
def test_core_columns_only_match_the_training_code(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 40)
    assert len(core_columns) == 224
    path = write_csv(tmp_path / "class.csv", core_columns, [[row[c] for c in core_columns] for row in rows])
    code, output, screen = run(clone, path)
    assert code == 0, screen
    assert "All 85 expected input columns found" in screen
    assert output == expected_counts(rows)
    assert sum(sum(row) > 0 for row in output) >= 2  # electron and next are flagged


def test_targets_never_influence_the_prediction(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 40)
    without = [c for c in core_columns if c not in TARGETS]
    rng = random.Random(1)
    path_a = write_csv(tmp_path / "a.csv", without, [[row[c] for c in without] for row in rows])
    path_b = write_csv(tmp_path / "b.csv", core_columns,
                       [[str(rng.randint(0, 9)) if c in TARGETS else row[c] for c in core_columns] for row in rows])
    assert run(clone, path_a)[1] == run(clone, path_b)[1]


# 2. Sizes and column layout -----------------------------------------------------------------
def test_single_row(clone, tmp_path, training_rows, core_columns):
    row = next(r for r in training_rows if r["purl"].startswith("pkg:npm/electron@"))
    code, output, _ = run(clone, write_csv(tmp_path / "one.csv", core_columns, [[row[c] for c in core_columns]]))
    assert code == 0 and output == [[0, 1, 0, 0, 0]]


def test_thousand_rows_keep_their_order(clone, tmp_path, training_rows, core_columns):
    rng = random.Random(2)
    rows = [rng.choice(training_rows) for _ in range(1000)]
    code, output, _ = run(clone, write_csv(tmp_path / "big.csv", core_columns, [[row[c] for c in core_columns] for row in rows]))
    assert code == 0 and output == expected_counts(rows)


def test_shuffled_and_extra_columns(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 20)
    columns = core_columns[:]
    random.Random(3).shuffle(columns)
    header = ["note", *columns, "Another Extra"]
    path = write_csv(tmp_path / "shuffled.csv", header, [["x", *[row[c] for c in columns], "y"] for row in rows])
    code, output, _ = run(clone, path)
    assert code == 0 and output == expected_counts(rows)


# 3. Format variations -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "delimiter, encoding, newline",
    [(",", "utf-8", "\r\n"), (";", "utf-8", "\n"), (",", "utf-8-sig", "\n"), (";", "utf-8-sig", "\r\n"), (",", "cp1252", "\n")],
)
def test_file_formats(clone, tmp_path, training_rows, core_columns, delimiter, encoding, newline):
    rows = sample(training_rows, 20)
    header = [f" {name.upper()} " for name in core_columns]  # case and spaces in column names
    body = [[row[c] for c in core_columns] for row in rows]
    if encoding == "cp1252":
        header.append("bemerkung")
        body = [[*values, "Größe café"] for values in body]  # not valid UTF-8 once saved as cp1252
    path = write_csv(tmp_path / "format.csv", header, body, delimiter, encoding, newline)
    code, output, screen = run(clone, path)
    assert code == 0, screen
    assert output == expected_counts(rows)
    assert ("Windows-1252" in screen) == (encoding == "cp1252")


def test_number_and_timestamp_spellings(clone, tmp_path, training_rows, core_columns):
    row = dict(next(r for r in training_rows if r["purl"].startswith("pkg:npm/electron@")))
    variant = dict(row)
    variant["unpacked_size"] = f"{float(row['unpacked_size']):.1E}" if row["unpacked_size"] else ""
    variant["dependency_count"] = f" {row['dependency_count']} "
    variant["ghsa_moderate_count"] = f"{row['ghsa_moderate_count']}.0"
    variant["package_created_at"] = row["package_created_at"][:19] + "+00:00"
    variant["published_at"] = row["published_at"][:19] + "Z"
    path = write_csv(tmp_path / "spell.csv", core_columns, [[variant[c] for c in core_columns]])
    code, output, _ = run(clone, path)
    assert code == 0 and output == [[0, 1, 0, 0, 0]]
    features, names = ft.build_features([variant], None, use_reference_features=False)
    assert not np.isnan(features[0, names.index("log_unpacked_size")])
    assert not np.isnan(features[0, names.index("days_created_to_published")])


# 4. Bad data --------------------------------------------------------------------------------
def test_empty_file_gives_header_only(clone, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    code, output, screen = run(clone, path)
    assert code == 0 and output == [] and "no data rows" in screen


def test_header_only(clone, tmp_path, core_columns):
    code, output, _ = run(clone, write_csv(tmp_path / "header.csv", core_columns, []))
    assert code == 0 and output == []


def test_bad_rows_do_not_crash(clone, tmp_path, training_rows, core_columns):
    good = next(r for r in training_rows if r["purl"].startswith("pkg:npm/electron@"))
    full = [good[c] for c in core_columns]
    text_in_numbers = ["lots" if c.startswith(("ghsa_", "package_downloads")) else good[c] for c in core_columns]
    nulls = ["NULL" if i % 2 else "NA" for i, _ in enumerate(core_columns)]
    rows = [full, full[:10], text_in_numbers, [""] * len(core_columns), nulls, full]
    code, output, screen = run(clone, write_csv(tmp_path / "bad.csv", core_columns, rows))
    assert code == 0, screen
    assert len(output) == 5  # the row of empty cells is skipped, with a notice
    assert "skipped 1 empty row" in screen
    assert output[0] == output[4] == [0, 1, 0, 0, 0]
    assert all(value >= 0 for row in output for value in row)
    assert "no usable feature values" in screen  # the NULL/NA row


def test_missing_purl_column(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 5)
    columns = [c for c in core_columns if c != "purl"]
    code, output, screen = run(clone, write_csv(tmp_path / "nopurl.csv", columns, [[row[c] for c in columns] for row in rows]))
    assert code == 0 and output == expected_counts(rows)


def test_missing_input_file_exits_2_without_output(clone, tmp_path):
    code, output, screen = run(clone, tmp_path / "does_not_exist.csv")
    assert code == 2 and output == [] and "cannot read" in screen


# 5. Broken model ----------------------------------------------------------------------------
def test_missing_model_file_gives_zeros_and_exit_1(clone, tmp_path, training_rows, core_columns):
    (clone / "models/pcom_entry_1/model.txt").unlink()
    rows = sample(training_rows, 5)
    code, output, screen = run(clone, write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows]))
    assert code == 1 and output == [ZERO] * 5 and "ALL predictions below are zeros" in screen


def test_mismatched_feature_list_gives_zeros_and_exit_1(clone, tmp_path, training_rows, core_columns):
    meta_path = clone / "models/pcom_entry_1/model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["feature_names"] = list(reversed(meta["feature_names"]))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    rows = sample(training_rows, 5)
    code, output, screen = run(clone, write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows]))
    assert code == 1 and output == [ZERO] * 5 and "disagree on the feature list" in screen


def test_model_with_crlf_line_endings_still_works(clone, tmp_path, training_rows, core_columns):
    """What git did on a Windows checkout with core.autocrlf: LightGBM crashed before the fix."""
    model = clone / "models/pcom_entry_1/model.txt"
    model.write_bytes(model.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert model.read_bytes().count(b"\r\n") > 1000
    rows = sample(training_rows, 10)
    code, output, screen = run(clone, write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows]))
    assert code == 0, screen
    assert output == expected_counts(rows)


def test_damaged_model_file_gives_zeros_not_a_crash(clone, tmp_path, training_rows, core_columns):
    model = clone / "models/pcom_entry_1/model.txt"
    text = model.read_bytes()
    model.write_bytes(text[: len(text) // 2])  # a truncated file, as from an interrupted copy
    rows = sample(training_rows, 5)
    code, output, screen = run(clone, write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows]))
    assert code == 1 and output == [ZERO] * 5 and "does not match model_meta.json" in screen


def inside_git_repo():
    if shutil.which("git") is None:
        return False
    return subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True).returncode == 0


def test_gitattributes_keeps_the_model_unconverted():
    if not inside_git_repo():
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "check-attr", "text", "--", "models/pcom_entry_1/model.txt"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.stdout.strip().endswith("text: unset")


def expected_values(rows):
    meta = json.loads((config.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    X, names = ft.build_features(rows, None, use_reference_features=False)
    booster = lgb.Booster(model_file=str(config.MODEL_DIR / "model.txt"))
    scores = np.nan_to_num(X[:, names.index("ghsa_total")]) + 0.5 * booster.predict(X)
    strategy = meta["counts"].get("default_float_strategy", "EV")
    return ct.predict_float(scores, rows, meta["counts"]["strategies"], strategy)


def test_float_table_is_printed_and_integers_stay_the_default(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 8)
    code, output, screen = run(clone, write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows]))
    assert code == 0 and output == expected_counts(rows)
    assert "Values for the form (float):" in screen
    table = screen.split("Values for the form (float):", 1)[1].splitlines()
    values = [[float(v) for v in line.split()[-5:]] for line in table if line.strip()[:1].isdigit()]
    assert np.allclose(values, expected_values(rows).round(3), atol=1e-9)
    assert all(row[4] == 0.0 for row in values)


def test_float_option_writes_decimals(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 8)
    path = write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows])
    output = tmp_path / "float.csv"
    result = subprocess.run([sys.executable, str(clone / "predict.py"), str(path), "-o", str(output), "--float"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    table = list(csv.reader(open(output, newline="", encoding="utf-8")))
    assert table[0] == TARGETS and len(table) == 9
    assert all("." in value for row in table[1:] for value in row)
    written = np.array([[float(value) for value in row] for row in table[1:]])
    assert (written >= 0).all()
    assert np.allclose(written, expected_values(rows), atol=1e-6)


def test_float_values_are_zero_when_the_model_is_missing(clone, tmp_path, training_rows, core_columns):
    (clone / "models/pcom_entry_1/model.txt").unlink()
    rows = sample(training_rows, 3)
    path = write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows])
    output = tmp_path / "float.csv"
    result = subprocess.run([sys.executable, str(clone / "predict.py"), str(path), "-o", str(output), "--float"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert [row for row in csv.reader(open(output, newline="", encoding="utf-8"))][1:] == [["0.000000"] * 5] * 3


# 6. Working directory -----------------------------------------------------------------------
def test_runs_from_another_folder(clone, tmp_path, training_rows, core_columns):
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    rows = sample(training_rows, 10)
    path = write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows])
    code, output, _ = run(clone, path, cwd=elsewhere)
    assert code == 0 and output == expected_counts(rows)


def test_strategy_option(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 10)
    path = write_csv(tmp_path / "in.csv", core_columns, [[row[c] for c in core_columns] for row in rows])
    code, output, _ = run(clone, path, "--strategy", "S0")
    assert code == 0 and output == [ZERO] * 10


def test_default_output_name(clone, tmp_path, training_rows, core_columns):
    rows = sample(training_rows, 3)
    path = write_csv(tmp_path / "round1.csv", core_columns, [[row[c] for c in core_columns] for row in rows])
    result = subprocess.run([sys.executable, str(clone / "predict.py"), str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and (tmp_path / "round1_predictions.csv").exists()


# 7. Fresh clone -----------------------------------------------------------------------------
def test_runtime_files_are_not_gitignored():
    if not inside_git_repo():
        pytest.skip("not a git checkout")
    for name in [*RUNTIME_FILES, "requirements.txt"]:
        assert (ROOT / name).exists(), name
        ignored = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT).returncode == 0
        assert not ignored, f"{name} is gitignored"
