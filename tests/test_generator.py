"""Tests for device and dtype selection (no model is loaded)."""

import pytest
import torch

from rag import generator


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, ("cuda", torch.float16)),
        (False, True, ("mps", torch.float16)),
        (False, False, ("cpu", torch.float32)),
    ],
)
def test_pick_device(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: tuple[str, torch.dtype]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert generator.pick_device() == expected
