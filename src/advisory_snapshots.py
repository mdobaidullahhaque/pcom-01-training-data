"""Download single-commit snapshots of the reviewed GitHub advisories into the cache.

Usage: python -m src.advisory_snapshots --only latest
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

import config

log = logging.getLogger("advisory_snapshots")


class SizeLimitExceeded(Exception):
    pass


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    _kernel32.GetFileSizeEx.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _INVALID_HANDLE = wintypes.HANDLE(-1).value

    def real_file_size(path):
        """Size from an open handle. Directory listings can show 0 for a file still being written."""
        file_read_attributes, share_all, open_existing = 0x80, 0x7, 3
        handle = _kernel32.CreateFileW(
            str(path), file_read_attributes, share_all, None, open_existing, 0, None
        )
        if handle == _INVALID_HANDLE:
            return os.path.getsize(path)
        try:
            size = ctypes.c_longlong()
            if not _kernel32.GetFileSizeEx(handle, ctypes.byref(size)):
                return os.path.getsize(path)
            return size.value
        finally:
            _kernel32.CloseHandle(handle)
else:
    def real_file_size(path):
        return os.path.getsize(path)


def folder_size(folder):
    """Total bytes in a folder, reading true sizes for files that may still be open."""
    total = 0
    for current, _, names in os.walk(folder):
        for name in names:
            try:
                total += real_file_size(Path(current) / name)
            except OSError:
                pass  # git renames and removes temporary files while we scan
    return total


def kill_process_tree(process):
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        process.kill()
    process.wait()


def run_guarded(command, cwd, watched, limit_bytes, interval=3):
    """Run a command, killing it if the watched folder grows past the limit."""
    with tempfile.TemporaryFile() as captured:
        process = subprocess.Popen(command, cwd=cwd, stdout=captured, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(interval)
            size = folder_size(watched)
            log.info("%s: %.1f MB", " ".join(map(str, command[:2])), size / 1024**2)
            if size > limit_bytes:
                kill_process_tree(process)
                raise SizeLimitExceeded(f"{watched} reached {size / 1024**2:.1f} MB")
        captured.seek(0)
        output = captured.read().decode(errors="replace")
    if process.returncode != 0:
        raise RuntimeError(f"{command} failed with {process.returncode}: {output}")
    size = folder_size(watched)
    if size > limit_bytes:
        raise SizeLimitExceeded(f"{watched} reached {size / 1024**2:.1f} MB")
    return output


def find_commit(until=None):
    params = {"per_page": 1}
    if until:
        params["until"] = until.strftime("%Y-%m-%dT%H:%M:%SZ")
    response = requests.get(config.ADVISORY_COMMITS_API, params=params, timeout=60)
    response.raise_for_status()
    commit = response.json()[0]
    return {"sha": commit["sha"], "date": commit["commit"]["committer"]["date"]}


def load_snapshots():
    """Resolve the three commits once, so reruns keep using the same 'latest'."""
    path = config.ADVISORY_SNAPSHOTS_JSON
    if path.exists():
        snapshots = json.loads(path.read_text(encoding="utf-8"))
    else:
        t0 = config.parse_timestamp(config.PREDICTION_DATES[0])
        snapshots = {
            "latest": find_commit(),
            "t0": find_commit(t0),
            "target_end": find_commit(config.target_window(t0)[1]),
        }
        path.write_text(json.dumps(snapshots, indent=2) + "\n", encoding="utf-8")
    for name, snapshot in snapshots.items():
        log.info("snapshot %s: %s (%s)", name, snapshot["sha"], snapshot["date"])
    return snapshots


def git(folder, *args):
    result = subprocess.run(["git", *args], cwd=folder, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def fetch_snapshot(name, sha, limit_bytes):
    folder = config.ADVISORY_SNAPSHOT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    git(folder, "init", "-q")
    # Everything below must act on the snapshot's own repo, never on the project repo.
    if Path(git(folder, "rev-parse", "--show-toplevel")).resolve() != folder.resolve():
        raise RuntimeError(f"{folder} is not its own git repository")
    if "origin" not in git(folder, "remote"):
        git(folder, "remote", "add", "origin", config.ADVISORY_DB_URL)
    git(folder, "sparse-checkout", "set", config.ADVISORY_SUBDIR)
    fetch = ["git", "fetch", "--depth", "1", "--filter=blob:none", "origin", sha]
    run_guarded(fetch, folder, folder, limit_bytes)
    run_guarded(["git", "checkout", "-q", sha], folder, folder, limit_bytes)
    if git(folder, "rev-parse", "HEAD") != sha:
        raise RuntimeError(f"{name}: HEAD is not {sha}")
    files = sum(1 for _ in (folder / config.ADVISORY_SUBDIR).rglob("*.json"))
    return folder_size(folder), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=["latest", "t0", "target_end"], required=True)
    args = parser.parse_args()

    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "advisory_snapshots.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    snapshots = load_snapshots()
    for name in args.only:
        started = time.time()
        try:
            size, files = fetch_snapshot(name, snapshots[name]["sha"], config.SNAPSHOT_SIZE_LIMIT_BYTES)
        except SizeLimitExceeded as error:
            log.error("STOPPED %s: %s", name, error)
            sys.exit(1)
        log.info(
            "DONE %s: %.1f MB on disk, %d advisory files, %.0f s",
            name, size / 1024**2, files, time.time() - started,
        )


if __name__ == "__main__":
    main()
