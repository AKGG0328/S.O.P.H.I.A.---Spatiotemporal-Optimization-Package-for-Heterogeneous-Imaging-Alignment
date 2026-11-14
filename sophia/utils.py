"""
Utility functions for SOPHIA file discovery and data handling.
"""
from pathlib import Path


def find_run_file(directory: Path, prefix: str, suffix: str, run: str | int) -> Path:
    """
    Locates a file matching either zero-padded (run-01) or unpadded (run-1) indexing.
    """
    run_int = int(run)
    candidates = [
        directory / f"{prefix}_run-{run_int:02d}_{suffix}",
        directory / f"{prefix}_run-{run_int}_{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"File not found in {directory}.\nTried:\n - {candidates[0]}\n - {candidates[1]}"
    )