import sys
import time

import pytest

from src.advisory_snapshots import SizeLimitExceeded, folder_size, run_guarded

MB = 1024**2


def test_folder_size_sees_a_file_that_is_still_open(tmp_path):
    with open(tmp_path / "tmp_pack", "wb") as growing:
        growing.write(b"x" * (5 * MB))
        growing.flush()
        assert folder_size(tmp_path) == 5 * MB


def test_guard_kills_a_process_that_goes_over_the_limit(tmp_path):
    marker = tmp_path / "finished"
    child = (
        "import time\n"
        "f = open('tmp_pack', 'wb')\n"
        "f.write(b'x' * (5 * 1024 * 1024)); f.flush()\n"
        "time.sleep(30)\n"
        f"open(r'{marker}', 'w').close()\n"
    )
    started = time.time()
    with pytest.raises(SizeLimitExceeded):
        run_guarded([sys.executable, "-c", child], tmp_path, tmp_path, MB, interval=0.5)
    assert time.time() - started < 20
    assert not marker.exists()


def test_guard_returns_when_under_the_limit(tmp_path):
    output = run_guarded([sys.executable, "-c", "print('ok')"], tmp_path, tmp_path, MB, interval=0.1)
    assert "ok" in output
