"""`image-search index` holds Windows awake for the run: on Modern Standby
laptops a display timeout IS standby, which suspends the indexer for hours."""

import ctypes
import sys

from image_search import cli


def test_keep_awake_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli.keep_awake() is False


def test_keep_awake_holds_system_and_display_on_windows(monkeypatch):
    calls = []

    class Kernel32:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)
            return 0x80000000  # the previous state; 0 would mean failure

    class WinDLL:
        kernel32 = Kernel32()

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", WinDLL(), raising=False)
    assert cli.keep_awake() is True
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    assert calls == [0x80000000 | 0x1 | 0x2]


def test_index_accepts_allow_sleep_opt_out():
    assert cli.build_parser().parse_args(["index", "--allow-sleep"]).allow_sleep is True
    assert cli.build_parser().parse_args(["index"]).allow_sleep is False
