"""A fake competition server ("Compy") for testing the /predict flow end to end. Standard library only.

It serves an input CSV for download and receives the multipart upload, then checks the uploaded
prediction.csv: exactly 1 row, exactly the 5 target headers in order, every value a plain
non-negative number (no NaN, no empty cells).

  GET  /input.csv  -> the CSV given with --csv
  POST /upload     <- multipart field "predictions"; saved as <out>/prediction.csv; answers JSON

Typical use (service already running on port 8000):
  python service/fake_compy.py --csv tests/data/round1_input.csv --trigger http://localhost:8000
This starts the server on port 9000, POSTs /predict to the service, waits for the upload, prints
the saved file, the checks and the timing, and exits (0 = valid upload, 1 = invalid or none).
Without --trigger it just serves until Ctrl+C.
"""
import argparse
import csv
import io
import json
import math
import sys
import threading
import time
import urllib.request
from email.parser import BytesParser
from email.policy import default as default_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TARGET_COLUMNS = [
    "target_ghsa_low_count_90d",
    "target_ghsa_moderate_count_90d",
    "target_ghsa_high_count_90d",
    "target_ghsa_critical_count_90d",
    "target_ghsa_unknown_count_90d",
]


def check_prediction_csv(text):
    """Return a list of problems with an uploaded prediction CSV (empty list = valid)."""
    try:
        records = list(csv.reader(io.StringIO(text)))
    except csv.Error as error:
        return [f"not a CSV: {error}"]
    records = [record for record in records if record]
    if not records:
        return ["empty file"]
    problems = []
    if records[0] != TARGET_COLUMNS:
        problems.append(f"headers {records[0]} != {TARGET_COLUMNS}")
    if len(records) - 1 != 1:
        problems.append(f"{len(records) - 1} data rows, expected exactly 1")
    for record in records[1:]:
        if len(record) != len(TARGET_COLUMNS):
            problems.append(f"{len(record)} values, expected {len(TARGET_COLUMNS)}")
        for value in record:
            try:
                number = float(value)
            except ValueError:
                problems.append(f"not a number: {value!r}")
                continue
            if not math.isfinite(number) or number < 0:
                problems.append(f"not a finite non-negative number: {value!r}")
            elif "e" in value.lower():
                problems.append(f"exponent notation: {value!r}")
    return problems


def parse_multipart(content_type, body):
    """Return {field name: (filename, content type, bytes)} from a multipart/form-data body."""
    message = BytesParser(policy=default_policy).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not message.is_multipart():
        return {}
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = (part.get_filename(), part.get_content_type(), part.get_payload(decode=True) or b"")
    return fields


class FakeCompy:
    """Serves self.files ({path: bytes}) and records uploads to /upload. port=0 picks a free port."""

    def __init__(self, files, out_dir=None, host="127.0.0.1", port=0):
        self.files = dict(files)
        self.out_dir = Path(out_dir) if out_dir else None
        self.uploads = []
        self.upload_event = threading.Event()
        compy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                print(f"[fake_compy] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

            def reply(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in compy.files:
                    self.reply(200, compy.files[self.path], "text/csv")
                else:
                    self.reply(404, b"not found", "text/plain")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path != "/upload":
                    return self.reply(404, b"not found", "text/plain")
                fields = parse_multipart(self.headers.get("Content-Type", ""), body)
                if "predictions" not in fields:
                    problems = [f"no 'predictions' field (got {sorted(fields)})"]
                    record = {"filename": None, "content_type": None, "text": "", "problems": problems}
                else:
                    filename, content_type, data = fields["predictions"]
                    text = data.decode("utf-8", errors="replace")
                    problems = check_prediction_csv(text)
                    if filename != "prediction.csv":
                        problems.append(f"filename {filename!r} != 'prediction.csv'")
                    if content_type != "text/csv":
                        problems.append(f"content type {content_type!r} != 'text/csv'")
                    record = {"filename": filename, "content_type": content_type, "text": text, "problems": problems}
                    if compy.out_dir:
                        compy.out_dir.mkdir(parents=True, exist_ok=True)
                        (compy.out_dir / "prediction.csv").write_bytes(data)
                record["received_at"] = time.perf_counter()
                compy.uploads.append(record)
                compy.upload_event.set()
                answer = {"status": "received", "valid": not problems, "problems": problems}
                self.reply(200, json.dumps(answer).encode("utf-8"), "application/json")

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def trigger(service_url, public_url, wait, compy):
    """POST /predict to the service, wait for the upload, print the result. Returns an exit code."""
    payload = {
        "testdata_download_endpoint": f"{public_url}/input.csv",
        "prediction_upload_endpoint": f"{public_url}/upload",
    }
    request = urllib.request.Request(f"{service_url.rstrip('/')}/predict", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = response.read().decode("utf-8")
    except OSError as error:  # URLError, HTTPError, connection refused, timeout
        print(f"ERROR: POST {service_url}/predict failed: {error}. Is the service running?")
        return 1
    print(f"POST /predict -> {answer} ({time.perf_counter() - started:.2f}s)", flush=True)
    if not compy.upload_event.wait(wait):
        print(f"ERROR: no upload within {wait}s; check the service log (docker logs ml-api)")
        return 1
    upload = compy.uploads[-1]
    print(f"\nUpload received {upload['received_at'] - started:.2f}s after POST /predict")
    print(f"filename={upload['filename']} content_type={upload['content_type']}")
    print("--- prediction.csv ---")
    print(upload["text"].rstrip("\n"))
    print("----------------------")
    if upload["problems"]:
        print("INVALID:", *upload["problems"], sep="\n  ")
        return 1
    print("VALID: 1 row, 5 target headers, plain non-negative numbers")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fake competition server for testing /predict.")
    parser.add_argument("--csv", required=True, help="input CSV to serve at /input.csv")
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--out", default=".", help="folder for the received prediction.csv (default: current folder)")
    parser.add_argument("--public-url", help="base URL the service uses to reach this server "
                                             "(default http://localhost:PORT)")
    parser.add_argument("--trigger", metavar="SERVICE_URL", help="POST /predict to this service, wait, report, exit")
    parser.add_argument("--wait", type=float, default=60, help="seconds to wait for the upload (default 60)")
    args = parser.parse_args(argv)

    compy = FakeCompy({"/input.csv": Path(args.csv).read_bytes()}, args.out, args.host, args.port).start()
    public_url = (args.public_url or f"http://localhost:{compy.port}").rstrip("/")
    print(f"[fake_compy] serving {args.csv} at {public_url}/input.csv; uploads to {public_url}/upload "
          f"are saved in {Path(args.out).resolve()}", flush=True)
    try:
        if args.trigger:
            return trigger(args.trigger, public_url, args.wait, compy)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        compy.stop()


if __name__ == "__main__":
    sys.exit(main())
