from __future__ import annotations

from pathlib import Path


def test_batch_probe_has_executable_entrypoint() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "batch_probe.py").read_text(encoding="utf-8")

    assert (
        'if __name__ == "__main__":\n    raise SystemExit(main())' in source
    )
