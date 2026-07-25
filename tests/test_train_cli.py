import subprocess
import sys
from pathlib import Path


def test_train_cli_exposes_required_actions() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    for action in ("probe", "overfit-one-batch", "run", "resume-check"):
        assert action in result.stdout
