"""Checks the Windows COM worker-thread lifecycle without touching the desktop."""
from __future__ import annotations

import sys
from types import ModuleType

import pytest

from voiceloop.actions import ActionRegistry


def test_minimize_all_initializes_and_releases_worker_com(monkeypatch):
    events = []
    com = ModuleType("pythoncom")
    com.CoInitialize = lambda: events.append("initialize")
    com.CoUninitialize = lambda: events.append("release")
    parent = ModuleType("win32com")
    parent.__path__ = []
    client = ModuleType("win32com.client")

    class FakeShell:
        def MinimizeAll(self):
            events.append("minimize")

    def dispatch(name):
        assert name == "Shell.Application"
        events.append("dispatch")
        return FakeShell()

    client.Dispatch = dispatch
    parent.client = client
    monkeypatch.setitem(sys.modules, "pythoncom", com)
    monkeypatch.setitem(sys.modules, "win32com", parent)
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    message, data = ActionRegistry._minimize_all_windows_sync()
    assert "Zminimalizowałem" in message
    assert data["mode"] == "minimize_all"
    assert events == ["initialize", "dispatch", "minimize", "release"]


def test_minimize_all_releases_com_if_shell_fails(monkeypatch):
    events = []
    com = ModuleType("pythoncom")
    com.CoInitialize = lambda: events.append("initialize")
    com.CoUninitialize = lambda: events.append("release")
    parent = ModuleType("win32com")
    parent.__path__ = []
    client = ModuleType("win32com.client")

    def dispatch(_name):
        raise RuntimeError("COM shell failed")

    client.Dispatch = dispatch
    parent.client = client
    monkeypatch.setitem(sys.modules, "pythoncom", com)
    monkeypatch.setitem(sys.modules, "win32com", parent)
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    with pytest.raises(RuntimeError, match="COM shell failed"):
        ActionRegistry._minimize_all_windows_sync()
    assert events == ["initialize", "release"]
