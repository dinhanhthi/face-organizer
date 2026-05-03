"""Run manifest: track processed files and settings across face-grouper runs."""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

MANIFEST_FILENAME = ".face-grouper-run.json"
SCHEMA_VERSION = 2


@dataclass
class RunEntry:
    """Per-(inputs, output) run record stored inside the manifest."""
    app_version: str
    command: str
    run_date: str
    inputs: list[str]
    settings: dict
    stats: dict
    processed_files: list[str] = field(default_factory=list)


def _inputs_hash(inputs: list[Path]) -> str:
    """Stable hash of the resolved, sorted input paths — identifies a unique run context."""
    key = "\n".join(sorted(str(p.resolve()) for p in inputs))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_entry(output_dir: Path, inputs: list[Path]) -> RunEntry | None:
    """Load the run entry for this specific set of inputs, or None if not found.

    Raises ValueError if the manifest file exists but has an unknown schema_version.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    v = data.get("schema_version")
    if v != SCHEMA_VERSION:
        raise ValueError(f"Unknown manifest schema_version: {v}")
    entry_data = data.get("runs", {}).get(_inputs_hash(inputs))
    if entry_data is None:
        return None
    return RunEntry(
        app_version=entry_data["app_version"],
        command=entry_data["command"],
        run_date=entry_data["run_date"],
        inputs=entry_data["inputs"],
        settings=entry_data["settings"],
        stats=entry_data["stats"],
        processed_files=entry_data.get("processed_files", []),
    )


def save_entry(output_dir: Path, inputs: list[Path], entry: RunEntry) -> None:
    """Write (or update) the run entry for this set of inputs atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    tmp_path = output_dir / (MANIFEST_FILENAME + ".tmp")

    # Load existing manifest to preserve other runs' entries
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        runs = existing.get("runs", {}) if existing.get("schema_version") == SCHEMA_VERSION else {}
    else:
        runs = {}

    runs[_inputs_hash(inputs)] = asdict(entry)
    payload = {"schema_version": SCHEMA_VERSION, "runs": runs}

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, manifest_path)


def compare_settings(stored: dict, current: dict) -> list[str]:
    """Return human-readable drift messages for keys that changed between runs."""
    messages = []
    for key, current_val in current.items():
        if key not in stored:
            continue
        if stored[key] != current_val:
            messages.append(f"{key} changed: {stored[key]} → {current_val}")
    return messages


def filter_unprocessed(paths: list[Path], processed: set[str]) -> list[Path]:
    return [p for p in paths if str(p.resolve()) not in processed]


def build_processed_set(entry: RunEntry | None) -> set[str]:
    return set(entry.processed_files) if entry else set()
