"""File organization: group mode, rename mode, dry-run, and collision handling."""

import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


_MAX_COLLISION_ATTEMPTS = 9999


def collision_free_path(
    target_dir: Path, stem: str, suffix: str, taken: set[Path]
) -> Path:
    """Return a path in target_dir that neither exists on disk nor is in taken.

    If stem + suffix is taken, tries stem_1 + suffix, stem_2 + suffix, etc.
    The caller is responsible for adding the returned path to taken.
    """
    candidate = target_dir / (stem + suffix)
    if not candidate.exists() and candidate not in taken:
        return candidate
    for counter in range(1, _MAX_COLLISION_ATTEMPTS + 1):
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists() and candidate not in taken:
            return candidate
    raise RuntimeError(
        f"Could not find a free filename for '{stem}{suffix}' in {target_dir} "
        f"after {_MAX_COLLISION_ATTEMPTS} attempts."
    )


def organize(
    image_paths: list[Path],
    no_face_paths: list[Path],
    labels: list[int],
    label_map: dict[int, int],
    output_dir: Path,
    mode: str,
    dry_run: bool,
    name_map: dict[int, str] | None = None,
    start_index: int = 1,
) -> None:
    """Copy images into organised output according to mode.

    Group mode:  output/person_N/original_name.ext  (with collision suffix)
    Rename mode: output/person_N_img_M.ext  (or unknown_img_M.ext)

    Args:
        image_paths: Paths that have embeddings (in same order as labels).
        no_face_paths: Paths with no detected face.
        labels: DBSCAN labels, parallel to image_paths.
        label_map: Mapping from original label → 1-based person number.
        output_dir: Root output directory.
        mode: "group" or "rename".
        dry_run: If True, print table only — no file copies.
        name_map: Optional mapping from DBSCAN label → display name (e.g. "john").
            When provided, matched clusters use this name instead of "person_N".
    """
    _names = name_map or {}

    def _label_name(label: int) -> str:
        return _names.get(label, f"person_{label_map[label]}")

    # Build plan: list of (source, destination) pairs
    plan: list[tuple[Path, Path]] = []

    if mode == "group":
        # ---- Group mode ----
        # Track planned destinations to avoid silent overwrites when two source
        # files share the same name (e.g. vacation/IMG_001.jpg + birthday/IMG_001.jpg).
        planned: set[Path] = set()

        for img_path, label in zip(image_paths, labels):
            if label == -1 or label not in label_map:
                subdir = output_dir / "unknown"
            else:
                subdir = output_dir / _label_name(label)

            dest = collision_free_path(subdir, img_path.stem, img_path.suffix, planned)
            planned.add(dest)
            plan.append((img_path, dest))

        # No-face images go to unknown/
        for img_path in no_face_paths:
            subdir = output_dir / "unknown"
            dest = collision_free_path(subdir, img_path.stem, img_path.suffix, planned)
            planned.add(dest)
            plan.append((img_path, dest))

    else:
        # ---- Rename mode ----
        # Per-prefix counters; skip numbers already occupied on disk from prior runs.
        person_counters: dict[str, int] = {}
        rename_taken: set[Path] = set()

        def _next_rename_dest(prefix: str, suffix: str) -> Path:
            person_counters.setdefault(prefix, start_index - 1)
            while True:
                person_counters[prefix] += 1
                candidate = output_dir / f"{prefix}_img_{person_counters[prefix]}{suffix}"
                if not candidate.exists() and candidate not in rename_taken:
                    return candidate

        for img_path, label in zip(image_paths, labels):
            prefix = (
                "unknown"
                if (label == -1 or label not in label_map)
                else _label_name(label)
            )
            dest = _next_rename_dest(prefix, img_path.suffix.lower())
            rename_taken.add(dest)
            plan.append((img_path, dest))

        for img_path in no_face_paths:
            dest = _next_rename_dest("unknown", img_path.suffix.lower())
            rename_taken.add(dest)
            plan.append((img_path, dest))

    if dry_run:
        _print_dry_run_table(plan)
        return

    # Execute copies — collect errors so a mid-run failure doesn't silently produce
    # partial output. All copyable files are attempted before reporting failures.
    errors: list[tuple[Path, str]] = []
    copied = 0
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            console.print(f"  [green]Copied[/green] {src.name} -> {dst}")
            copied += 1
        except OSError as exc:
            console.print(f"  [red]Failed[/red] {src}: {exc}")
            errors.append((src, str(exc)))

    if errors:
        console.print(
            f"\n[bold yellow]Done with errors.[/bold yellow] "
            f"{copied}/{len(plan)} file(s) copied — {len(errors)} failed:"
        )
        for src, msg in errors:
            console.print(f"  [red]✗[/red] {src}: {msg}")
        raise SystemExit(1)

    console.print(f"\n[bold green]Done.[/bold green] {copied} file(s) organised into {output_dir}")


def _print_dry_run_table(plan: list[tuple[Path, Path]]) -> None:
    """Print a rich table showing the planned source → destination mapping."""
    table = Table(title="Dry-run preview — no files will be copied")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Destination", style="green")

    for idx, (src, dst) in enumerate(plan, start=1):
        table.add_row(str(idx), str(src), str(dst))

    console.print(table)
    console.print(
        f"\n[bold yellow]Dry-run:[/bold yellow] {len(plan)} file(s) would be organised."
    )
