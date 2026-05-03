"""Video organisation: group by detected person, handle skipped and unknown videos."""

import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

from face_grouper.organizer import collision_free_path

console = Console()


def organize_videos(
    embedded_paths: list[Path],
    skipped_paths: list[Path],
    labels: list[int],
    label_map: dict[int, int],
    output_dir: Path,
    mode: str,
    dry_run: bool,
    name_map: dict[int, str] | None = None,
    start_index: int = 1,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Copy videos into organised output folders grouped by detected person.

    Each unique video is copied once per person detected in it. Videos where
    every detected face is an outlier (label -1) are placed in unknown/.
    Videos that were too long, unreadable, or had no faces are in skipped/.

    Args:
        embedded_paths: Video paths corresponding to each embedding, in the same
            order as labels. The same path may appear multiple times.
        skipped_paths: Videos that were too long, unreadable, or had zero faces.
        labels: DBSCAN labels parallel to embedded_paths.
        label_map: Mapping from original DBSCAN label → 1-based person number.
        output_dir: Root output directory.
        mode: "group" → person_N/ subfolders; "rename" → flat person_N_vid_M.ext files.
        dry_run: If True, print a table only — no file copies.
        name_map: Optional mapping from DBSCAN label → display name (e.g. "john").
            When provided, matched clusters use this name instead of "person_N".
    """
    _names = name_map or {}

    def _label_name(label: int) -> str:
        return _names.get(label, f"person_{label_map[label]}")

    # Build video_to_persons: {video_path: set of person labels (excluding -1)}
    video_to_persons: dict[Path, set[int]] = {}
    for video_path, label in zip(embedded_paths, labels):
        if video_path not in video_to_persons:
            video_to_persons[video_path] = set()
        if label != -1 and label in label_map:
            video_to_persons[video_path].add(label)

    # Build plan: list of (source, destination, reason) triples
    plan: list[tuple[Path, Path, str]] = []
    planned: set[Path] = set()

    if mode == "group":
        for video_path, persons in video_to_persons.items():
            if not persons:
                subdir = output_dir / "unknown"
                dest = collision_free_path(subdir, video_path.stem, video_path.suffix, planned)
                planned.add(dest)
                plan.append((video_path, dest, "unknown"))
            else:
                for label in sorted(persons):
                    subdir = output_dir / _label_name(label)
                    dest = collision_free_path(subdir, video_path.stem, video_path.suffix, planned)
                    planned.add(dest)
                    plan.append((video_path, dest, _label_name(label)))

        for video_path in skipped_paths:
            subdir = output_dir / "skipped"
            dest = collision_free_path(subdir, video_path.stem, video_path.suffix, planned)
            planned.add(dest)
            plan.append((video_path, dest, "skipped"))

    else:
        # ---- Rename mode ----
        person_counters: dict[str, int] = {}
        rename_taken: set[Path] = set()

        def _next_rename_dest(prefix: str, suffix: str) -> Path:
            person_counters.setdefault(prefix, start_index - 1)
            while True:
                person_counters[prefix] += 1
                candidate = output_dir / f"{prefix}_vid_{person_counters[prefix]}{suffix}"
                if not candidate.exists() and candidate not in rename_taken:
                    return candidate

        for video_path, persons in video_to_persons.items():
            if not persons:
                dest = _next_rename_dest("unknown", video_path.suffix.lower())
                rename_taken.add(dest)
                plan.append((video_path, dest, "unknown"))
            else:
                for label in sorted(persons):
                    dest = _next_rename_dest(_label_name(label), video_path.suffix.lower())
                    rename_taken.add(dest)
                    plan.append((video_path, dest, _label_name(label)))

        for video_path in skipped_paths:
            dest = _next_rename_dest("skipped", video_path.suffix.lower())
            rename_taken.add(dest)
            plan.append((video_path, dest, "skipped"))

    if dry_run:
        _print_dry_run_table(plan)
        return [], []

    # Execute copies
    errors: list[tuple[Path, str]] = []
    copied = 0
    for src, dst, reason in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            console.print(f"  [green]Copied[/green] {src.name} -> {dst}  [dim]({reason})[/dim]")
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
            console.print(f"  [red]x[/red] {src}: {msg}")
    else:
        console.print(f"\n[bold green]Done.[/bold green] {copied} file(s) organised into {output_dir}")

    failed_sources = {src for src, _ in errors}
    successes = sorted({src for src, _, _reason in plan if src not in failed_sources}, key=str)
    return successes, errors


def _print_dry_run_table(plan: list[tuple[Path, Path, str]]) -> None:
    """Print a rich table showing the planned source → destination mapping with reasons."""
    table = Table(title="Dry-run preview — no files will be copied")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Destination", style="green")
    table.add_column("Reason", style="yellow")

    for idx, (src, dst, reason) in enumerate(plan, start=1):
        table.add_row(str(idx), str(src), str(dst), reason)

    console.print(table)
    console.print(
        f"\n[bold yellow]Dry-run:[/bold yellow] {len(plan)} file(s) would be organised."
    )
