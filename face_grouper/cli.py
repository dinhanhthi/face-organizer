"""Command-line interface for face-grouper."""

from pathlib import Path

import click
from rich.console import Console

from face_grouper import __version__

console = Console()

_EPILOG = """
Examples:

  face-grouper group ./photos --output ./sorted

  face-grouper group ./photos --output ./sorted --mode rename

  face-grouper group ./photos --output ./sorted --mode rename --keep-originals

  face-grouper group img1.jpg img2.jpg img3.jpg --output ./sorted

  face-grouper group ./photos --output ./sorted --eps 0.45

  face-grouper group ./photos --output ./sorted --dry-run
"""


@click.group()
@click.version_option(__version__, prog_name="face-grouper")
def cli() -> None:
    """face-grouper: Group photos by person using offline face recognition.

    Detects faces in images, clusters them with DBSCAN, and organises the
    results into labelled folders or renamed flat files — all without any
    cloud API or internet connection.
    """


@cli.command(name="group", epilog=_EPILOG)
@click.argument(
    "inputs",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    metavar="INPUTS...",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory where organised images will be written.",
)
@click.option(
    "--mode",
    type=click.Choice(["group", "rename"], case_sensitive=False),
    default="group",
    show_default=True,
    help=(
        "Organisation mode. "
        "'group' copies files into person_N/ subfolders. "
        "'rename' copies files to the output root with person_N_img_M.ext names."
    ),
)
@click.option(
    "--keep-originals",
    is_flag=True,
    default=False,
    help=(
        "Accepted for compatibility. Originals are NEVER modified in any mode; "
        "this flag has no additional effect."
    ),
)
@click.option(
    "--backend",
    type=click.Choice(["dlib", "arcface"], case_sensitive=False),
    default="dlib",
    show_default=True,
    help=(
        "Face recognition backend. "
        "'dlib' uses face_recognition (fast, 128-D embeddings). "
        "'arcface' uses DeepFace+RetinaFace (more accurate, 512-D embeddings, "
        "downloads ~300 MB model on first run)."
    ),
)
@click.option(
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="hog",
    show_default=True,
    help=(
        "Face detection model. "
        "'hog' is fast but misses non-frontal or small faces. "
        "'cnn' is much more accurate but significantly slower (GPU recommended)."
    ),
)
@click.option(
    "--upsample",
    type=int,
    default=1,
    show_default=True,
    help=(
        "How many times to upsample the image before detection. "
        "Higher values find smaller faces but use more memory and time."
    ),
)
@click.option(
    "--eps",
    type=float,
    default=0.5,
    show_default=True,
    help=(
        "DBSCAN epsilon: maximum distance between two face embeddings to be "
        "considered the same person. Lower values = stricter grouping."
    ),
)
@click.option(
    "--min-samples",
    type=int,
    default=2,
    show_default=True,
    help=(
        "Minimum number of photos needed to form a person cluster. "
        "A person appearing in fewer photos than this threshold will be placed in unknown/. "
        "Set to 1 to keep every detected face as its own group."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview the planned file operations without copying anything.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help=(
        "Print pairwise embedding distance statistics after detection. "
        "Use this to find a good --eps value for your photo set."
    ),
)
def group_command(
    inputs: tuple[Path, ...],
    output: Path,
    mode: str,
    keep_originals: bool,
    backend: str,
    model: str,
    upsample: int,
    eps: float,
    min_samples: int,
    dry_run: bool,
    debug: bool,
) -> None:
    """Scan INPUTS for images, detect faces, cluster by person, and organise output.

    INPUTS can be directories (scanned recursively) or individual image files,
    or a mix of both. Supported formats: .jpg .jpeg .png .webp .bmp
    """
    from face_grouper.clusterer import cluster_embeddings, embedding_distance_stats, relabel_by_frequency
    from face_grouper.detector import detect_faces
    from face_grouper.organizer import organize
    from face_grouper.scanner import scan_images

    # --- 1. Collect inputs ---
    if not inputs:
        raise click.UsageError("No inputs provided. Pass files or directories.")

    console.print(f"[bold]face-grouper[/bold] v{__version__}")
    console.print(f"Scanning {len(inputs)} input(s)...")

    image_paths = scan_images(list(inputs))

    if not image_paths:
        raise click.UsageError("No supported image files found in the provided inputs.")

    console.print(f"Found [cyan]{len(image_paths)}[/cyan] image(s).")

    # --- 2. Overwrite check ---
    if (
        not dry_run
        and output.exists()
        and output.is_dir()
        and any(output.iterdir())
    ):
        click.confirm(
            f"Output directory '{output}' already exists and is not empty. "
            "Continue and merge into it?",
            abort=True,
        )

    # --- 3. Detect faces ---
    console.print("\nStep 1/3 — Face detection")
    embeddings, embedded_paths, no_face_paths = detect_faces(
        image_paths, model=model, upsample=upsample, backend=backend
    )

    console.print(
        f"  [green]{len(embedded_paths)}[/green] image(s) with faces, "
        f"[yellow]{len(no_face_paths)}[/yellow] without."
    )

    # --- 4. Cluster ---
    console.print("\nStep 2/3 — Clustering")
    labels: list[int] = []
    label_map: dict[int, int] = {}

    # ArcFace (insightface) produces unit-normalized 512-D vectors — cosine distance
    # is the natural metric. dlib produces 128-D embeddings best compared with euclidean.
    metric = "cosine" if backend.lower() == "arcface" else "euclidean"
    console.print(f"  [dim]metric={metric}[/dim]")

    if embeddings:
        if debug:
            stats = embedding_distance_stats(embeddings, metric=metric)
            if stats:
                console.print(
                    f"  [dim]Distance stats ({metric}) — "
                    f"min: {stats['min']:.3f}  "
                    f"p25: {stats['p25']:.3f}  "
                    f"median: {stats['median']:.3f}  "
                    f"p75: {stats['p75']:.3f}  "
                    f"max: {stats['max']:.3f}[/dim]"
                )
                console.print(
                    f"  [dim]Suggested --eps: just below the gap between p25 "
                    f"(same person) and p75 (different people).[/dim]"
                )

        labels = cluster_embeddings(embeddings, eps=eps, min_samples=min_samples, metric=metric)
        label_map = relabel_by_frequency(labels)
        unique_persons = len(label_map)
        outliers = labels.count(-1)
        console.print(
            f"  [green]{unique_persons}[/green] person group(s) found, "
            f"[yellow]{outliers}[/yellow] outlier(s)."
        )

        if unique_persons == 1 and len(embeddings) > 1:
            suggested = round(max(0.1, eps - 0.1), 2)
            console.print(
                f"  [yellow]Tip:[/yellow] Only 1 group found — try a lower "
                f"[bold]--eps {suggested}[/bold], or run with [bold]--debug[/bold] "
                f"to inspect distance distribution."
            )
    else:
        console.print("  No embeddings to cluster — all images will go to unknown/.")

    # --- 5. Organise ---
    console.print(f"\nStep 3/3 — Organising (mode={mode}{'  DRY-RUN' if dry_run else ''})")
    organize(
        image_paths=embedded_paths,
        no_face_paths=no_face_paths,
        labels=labels,
        label_map=label_map,
        output_dir=output,
        mode=mode,
        dry_run=dry_run,
    )
