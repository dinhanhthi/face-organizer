"""Run manifest: track processed files and settings across face-grouper runs."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

MANIFEST_FILENAME = ".face-grouper-run.json"
SCHEMA_VERSION = 1


@dataclass
class RunManifest:
    schema_version: int
    app_version: str
    command: str
    run_date: str
    inputs: list[str]
    settings: dict
    stats: dict
    processed_files: list[str] = field(default_factory=list)


def load_manifest(output_dir: Path) -> RunManifest | None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    v = data.get("schema_version")
    if v != SCHEMA_VERSION:
        raise ValueError(f"Unknown manifest schema_version: {v}")
    return RunManifest(
        schema_version=data["schema_version"],
        app_version=data["app_version"],
        command=data["command"],
        run_date=data["run_date"],
        inputs=data["inputs"],
        settings=data["settings"],
        stats=data["stats"],
        processed_files=data.get("processed_files", []),
    )


def save_manifest(output_dir: Path, manifest: RunManifest) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / (MANIFEST_FILENAME + ".tmp")
    manifest_path = output_dir / MANIFEST_FILENAME
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
    os.replace(tmp_path, manifest_path)


def compare_settings(stored: dict, current: dict) -> list[str]:
    messages = []
    for key, current_val in current.items():
        if key not in stored:
            continue
        stored_val = stored[key]
        if stored_val != current_val:
            messages.append(f"{key} changed: {stored_val} → {current_val}")
    return messages


def filter_unprocessed(paths: list[Path], processed: set[str]) -> list[Path]:
    return [p for p in paths if str(p.resolve()) not in processed]


def build_processed_set(manifest: RunManifest | None) -> set[str]:
    return set(manifest.processed_files) if manifest else set()
