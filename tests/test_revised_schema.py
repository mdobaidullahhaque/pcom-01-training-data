"""Homework 7: the revised PCOM-01 schema (docs/schema/PCOM-01_Revised_Schema_Public.xlsx).

The xlsx is read with the standard library only (an xlsx is a zip of XML files). Field names are
column C of each schema tab; "M0 ... M11" ranges (an _M0 name, a "…" row, an _M11 name) are
expanded to all 12 months.
"""
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "docs" / "schema" / "PCOM-01_Revised_Schema_Public.xlsx"
REVISED_INPUT = ROOT / "tests" / "data" / "revised_schema_input.csv"

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
_M = "{%s}" % _NS["m"]
_ELLIPSIS = {"…", "...", "..", "⋯"}


def read_sheet_column(path, sheet_name, column="C"):
    """Text of one column of one sheet, top to bottom (empty cells skipped)."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(text.text or "" for text in item.iter(_M + "t")))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {relation.get("Id"): relation.get("Target") for relation in relations}
        for sheet in workbook.find("m:sheets", _NS):
            if sheet.get("name") == sheet_name:
                target = targets[sheet.get("{%s}id" % _NS["r"])].lstrip("/")
                sheet_xml = archive.read(target if target.startswith("xl/") else "xl/" + target)
                break
        else:
            raise KeyError(f"no sheet named {sheet_name!r} in {path}")
    values = []
    for cell in ET.fromstring(sheet_xml).iter(_M + "c"):
        if re.sub(r"\d", "", cell.get("r", "")) != column:
            continue
        kind, value = cell.get("t"), cell.find("m:v", _NS)
        if kind == "s" and value is not None:
            text = shared[int(value.text)]
        elif kind == "inlineStr":
            text = "".join(part.text or "" for part in cell.iter(_M + "t"))
        else:
            text = value.text if value is not None else ""
        if text and text.strip():
            values.append(text.strip())
    return values


def field_names(sheet_name, path=XLSX):
    """Field names of a schema tab, in order, with M0 ... M11 ranges expanded."""
    cells = read_sheet_column(path, sheet_name)
    cells = cells[cells.index("Field name") + 1:]
    names = []
    for index, cell in enumerate(cells):
        if cell in _ELLIPSIS:
            before, after = names[-1] if names else "", cells[index + 1] if index + 1 < len(cells) else ""
            if not (before.endswith("_M0") and after == before[:-1] + "11"):
                raise ValueError(f"unexpected range {before!r} … {after!r} in {sheet_name}")
            names.extend(f"{before[:-1]}{month}" for month in range(1, 11))
        else:
            names.append(cell)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate field names in {sheet_name}")
    return names


needs_xlsx = pytest.mark.skipif(not XLSX.exists(), reason="revised schema xlsx not present")


@needs_xlsx
def test_schema_tabs_have_the_expected_shape():
    inputs = field_names("Input_Data_Schema")
    assert inputs[0] == "purl"
    assert len(inputs) == 219
    assert field_names("Target_Data_Schema") == field_names("Prediction_Schema")
    assert len(field_names("Prediction_Schema")) == 5


@needs_xlsx
def test_service_output_headers_match_the_prediction_schema():
    """Fails if the uploaded CSV's headers ever differ from Prediction_Schema in the xlsx."""
    from service import main as service
    from service.fake_compy import TARGET_COLUMNS as FAKE_COMPY_COLUMNS
    import predict

    expected = field_names("Prediction_Schema")
    output = service.run_model(pd.read_csv(REVISED_INPUT, dtype=str))
    assert list(output.columns) == expected
    assert service.to_csv_text(output).splitlines()[0].split(",") == expected
    assert list(service.TARGET_COLUMNS) == expected
    assert list(predict.TARGET_COLUMNS) == expected
    assert list(FAKE_COMPY_COLUMNS) == expected


@needs_xlsx
def test_target_names_match_the_prediction_names():
    import predict

    assert field_names("Target_Data_Schema") == list(predict.TARGET_COLUMNS)


@needs_xlsx
def test_every_model_input_column_is_in_the_revised_input_schema():
    """A renamed input column would silently turn a feature into NaN."""
    from src.model import features as ft

    missing = sorted(set(ft.INPUT_COLUMNS) - set(field_names("Input_Data_Schema")))
    assert missing == []


@needs_xlsx
def test_revised_input_fixture_has_exactly_the_input_schema_columns():
    with open(REVISED_INPUT, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == field_names("Input_Data_Schema")


def test_run_model_on_revised_input_gives_one_valid_row():
    from service import main as service
    from service.fake_compy import check_prediction_csv

    output = service.run_model(pd.read_csv(REVISED_INPUT, dtype=str))
    assert output.shape == (1, 5)
    assert (output.to_numpy(dtype=float) >= 0).all()
    assert check_prediction_csv(service.to_csv_text(output)) == []
