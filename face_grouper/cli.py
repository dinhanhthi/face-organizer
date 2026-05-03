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

  face-grouper group img1.jpg img2.jpg img3.jpg --output ./sorted

  face-grouper group ./photos --output ./sorted --eps 0.45

  face-grouper group ./photos --output ./sorted --dry-run
"""

_VIDEO_EPILOG = """
Examples:

  fgroup video ./clips --output ./sorted

  fgroup video ./clips --output ./sorted --mode rename

  fgroup video ./clips --output ./sorted --backend arcface

  fgroup video ./clips --output ./sorted --max-duration 30

  fgroup video ./clips --output ./sorted --dry-run

  fgroup video clip1.mp4 clip2.mp4 --output ./sorted

Note: requires opencv-python — install with:  pip install 'face-grouper[video]'
"""


@click.group()
@click.version_option(__version__, prog_name="face-grouper")
def cli() -> None:
    """face-grouper: Group photos by person using offline face recognition.

    Detects faces in images, clusters them with DBSCAN, and organises the
    results into labelled folders or renamed flat files — all without any
    cloud API or internet connection.
    """


def _resolve_reference_names(
    embeddings: list,
    labels: list[int],
    label_map: dict[int, int],
    reference_embeddings: dict,
    metric: str,
    eps: float,
) -> dict[int, str]:
    """Map DBSCAN cluster labels to reference names via greedy nearest-neighbour matching.

    For each reference name (which may have multiple embeddings from multiple photos),
    finds the cluster whose nearest member is within eps distance of any reference photo.
    Greedy assignment (sorted by distance ascending) ensures each cluster and each
    reference name is used at most once.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    if not reference_embeddings or not embeddings:
        return {}

    cluster_embs: dict[int, list] = {}
    for emb, label in zip(embeddings, labels):
        if label == -1:
            continue
        cluster_embs.setdefault(label, []).append(emb)

    if not cluster_embs:
        return {}

    dist_fn = cosine_distances if metric == "cosine" else euclidean_distances

    candidates: list[tuple[float, str, int]] = []
    for ref_name, ref_embs in reference_embeddings.items():
        if not ref_embs:
            continue
        ref_matrix = np.array(ref_embs)  # shape: (n_ref_photos, dim)
        for label, embs in cluster_embs.items():
            dists = dist_fn(ref_matrix, np.array(embs))  # shape: (n_ref_photos, n_cluster_embs)
            candidates.append((float(dists.min()), ref_name, label))

    candidates.sort()

    assigned_refs: set[str] = set()
    assigned_labels: set[int] = set()
    result: dict[int, str] = {}

    for min_dist, ref_name, label in candidates:
        if min_dist > eps:
            break
        if ref_name in assigned_refs or label in assigned_labels:
            continue
        result[label] = ref_name
        assigned_refs.add(ref_name)
        assigned_labels.add(label)

    return result


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
@click.option(
    "--ref-dir",
    "--reference-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory of named reference images. Clusters matching a reference are named "
        "after the image stem: e.g. 'john.jpg' → 'john/' in group mode. "
        "Multiple photos per person are supported via a _N suffix: "
        "'john_1.jpg', 'john_2.jpg', ... all map to 'john'."
    ),
)
@click.option(
    "--no-multi-export",
    is_flag=True,
    default=False,
    help=(
        "Only use the largest detected face per photo (legacy behaviour). "
        "By default every detected face in a photo is exported independently, "
        "so a photo with two people lands in both person folders."
    ),
)
@click.option(
    "--start-index",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Starting index for file counters in rename mode (e.g. --start-index 10 "
        "produces person_N_img_10.ext, person_N_img_11.ext, ...). "
        "Ignored in group mode."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip files already processed in a previous run. Warns if settings differ.",
)
def group_command(
    inputs: tuple[Path, ...],
    output: Path,
    mode: str,
    backend: str,
    model: str,
    upsample: int,
    eps: float,
    min_samples: int,
    dry_run: bool,
    debug: bool,
    ref_dir: Path | None,
    no_multi_export: bool,
    start_index: int,
    resume: bool,
) -> None:
    """Scan INPUTS for images, detect faces, cluster by person, and organise output.

    INPUTS can be directories (scanned recursively) or individual image files,
    or a mix of both. Supported formats: .jpg .jpeg .png .webp .bmp
    """
    from face_grouper.clusterer import cluster_embeddings, embedding_distance_stats, relabel_by_frequency
    from face_grouper.detector import detect_faces, detect_reference_faces
    from face_grouper.organizer import organize
    from face_grouper.scanner import scan_images

    # --- 1. Collect inputs ---
    if not inputs:
        raise click.UsageError("No inputs provided. Pass files or directories.")

    console.print(f"[bold]face-grouper[/bold] v{__version__}")
    console.print(f"Scanning {len(inputs)} input(s)...")

    image_paths = scan_images(list(inputs))

    # Exclude files that live inside the output directory so that a previous run's
    # output is never re-processed (which would silently duplicate every face).
    output_resolved = output.resolve()
    pre_exclude = len(image_paths)
    image_paths = [p for p in image_paths if not p.is_relative_to(output_resolved)]
    if len(image_paths) < pre_exclude:
        console.print(
            f"  [dim]Excluded {pre_exclude - len(image_paths)} file(s) already inside "
            f"the output directory (from a previous run).[/dim]"
        )

    # Load previous manifest for --resume
    import datetime
    from face_grouper.run_tracker import (
        RunManifest,
        build_processed_set,
        compare_settings,
        filter_unprocessed,
        load_manifest,
        save_manifest,
    )
    prior_manifest: RunManifest | None = None
    if resume:
        prior_manifest = load_manifest(output)
        if prior_manifest is None:
            console.print("[dim]No previous run found, starting fresh.[/dim]")
        else:
            drift = compare_settings(prior_manifest.settings, {
                "backend": backend, "model": model, "eps": eps,
                "min_samples": min_samples, "mode": mode, "upsample": upsample,
                "no_multi_export": no_multi_export, "start_index": start_index,
                "ref_dir": str(ref_dir) if ref_dir is not None else None,
            })
            for msg in drift:
                console.print(f"  [yellow]Warning:[/yellow] {msg}")
            processed_set = build_processed_set(prior_manifest)
            before = len(image_paths)
            image_paths = filter_unprocessed(image_paths, processed_set)
            skipped_count = before - len(image_paths)
            if skipped_count:
                console.print(f"  [dim]Resuming: skipping {skipped_count} already-processed file(s).[/dim]")
            if not image_paths:
                console.print("[bold green]All inputs already processed.[/bold green]")
                return

    if not image_paths:
        raise click.UsageError("No supported image files found in the provided inputs.")

    console.print(f"Found [cyan]{len(image_paths)}[/cyan] image(s).")

    # --- 1b. Load reference embeddings ---
    import re as _re
    reference_embeddings: dict = {}
    if ref_dir is not None:
        ref_paths = scan_images([ref_dir])
        if ref_paths:
            # Exclude reference images from the main scan to avoid double-processing
            ref_paths_set = set(ref_paths)
            before = len(image_paths)
            image_paths = [p for p in image_paths if p not in ref_paths_set]
            if len(image_paths) < before:
                console.print(f"  [dim]Excluded {before - len(image_paths)} reference image(s) from main scan.[/dim]")

            # Validate reserved names from filenames before running expensive detection
            potential_names = {(_re.sub(r"_\d+$", "", p.stem) or p.stem) for p in ref_paths}
            reserved_conflicts = {
                name for name in potential_names
                if name == "unknown" or _re.match(r"^person_\d+$", name)
            }
            if reserved_conflicts:
                raise click.UsageError(
                    f"Reference image name(s) conflict with reserved names: "
                    f"{', '.join(sorted(reserved_conflicts))}. "
                    f"Rename them to avoid 'unknown' and 'person_N' patterns."
                )

            console.print(f"Loading reference faces from [cyan]{len(ref_paths)}[/cyan] image(s) in {ref_dir}...")
            reference_embeddings = detect_reference_faces(
                ref_paths, backend=backend, model=model, upsample=upsample
            )

        if not reference_embeddings:
            console.print(
                "[yellow]Warning:[/yellow] No valid reference faces found — "
                "clusters will use default person_N naming."
            )
        else:
            total_photos = sum(len(v) for v in reference_embeddings.values())
            names = ", ".join(f"[bold]{n}[/bold]" for n in sorted(reference_embeddings))
            console.print(
                f"Loaded [cyan]{len(reference_embeddings)}[/cyan] person(s) from "
                f"[cyan]{total_photos}[/cyan] reference photo(s): {names}"
            )

    # --- 2. Overwrite check ---
    if (
        not dry_run
        and not resume
        and output.exists()
        and output.is_dir()
        and any(output.iterdir())
    ):
        click.confirm(
            f"Output directory '{output}' already exists and is not empty. "
            "Existing files will NOT be overwritten — new files will be added with "
            "incremented names if there are conflicts. Continue?",
            abort=True,
        )

    # --- 3. Detect faces ---
    console.print("\nStep 1/3 — Face detection")
    embeddings, embedded_paths, no_face_paths = detect_faces(
        image_paths, model=model, upsample=upsample, backend=backend,
        multi_face=not no_multi_export,
    )

    n_unique = len(set(embedded_paths))
    console.print(
        f"  [green]{len(embedded_paths)}[/green] face(s) across "
        f"[green]{n_unique}[/green] image(s) with faces, "
        f"[yellow]{len(no_face_paths)}[/yellow] without."
    )
    if len(embedded_paths) > n_unique:
        console.print(
            "  [dim](multi-face export ON — use --no-multi-export to disable)[/dim]"
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
        name_map = _resolve_reference_names(embeddings, labels, label_map, reference_embeddings, metric, eps)
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
        if name_map:
            console.print(
                f"  Reference matches: "
                + ", ".join(f"person_{label_map[lbl]} → [bold]{name}[/bold]" for lbl, name in name_map.items())
            )
    else:
        console.print("  No embeddings to cluster — all images will go to unknown/.")
        name_map = {}

    # --- 5. Organise ---
    console.print(f"\nStep 3/3 — Organising (mode={mode}{'  DRY-RUN' if dry_run else ''})")
    successes, org_errors = organize(
        image_paths=embedded_paths,
        no_face_paths=no_face_paths,
        labels=labels,
        label_map=label_map,
        output_dir=output,
        mode=mode,
        dry_run=dry_run,
        name_map=name_map or None,
        start_index=start_index,
    )

    if not dry_run:
        prior_processed = build_processed_set(prior_manifest)
        new_processed = {str(p.resolve()) for p in successes}
        all_processed = sorted(prior_processed | new_processed)
        manifest = RunManifest(
            schema_version=1,
            app_version=__version__,
            command="group",
            run_date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            inputs=[str(Path(p).resolve()) for p in inputs],
            settings={
                "backend": backend, "model": model, "eps": eps,
                "min_samples": min_samples, "mode": mode, "upsample": upsample,
                "no_multi_export": no_multi_export, "start_index": start_index,
                "ref_dir": str(ref_dir) if ref_dir is not None else None,
            },
            stats={
                "n_images_scanned": len(image_paths),
                "n_faces_detected": len(embedded_paths),
                "n_persons_found": len(label_map),
                "n_no_face": len(no_face_paths),
            },
            processed_files=all_processed,
        )
        save_manifest(output, manifest)
    if org_errors:
        raise SystemExit(1)


@cli.command(name="video", epilog=_VIDEO_EPILOG)
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
    help="Output directory where organised videos will be written.",
)
@click.option(
    "--mode",
    type=click.Choice(["group", "rename"], case_sensitive=False),
    default="group",
    show_default=True,
    help=(
        "Organisation mode. "
        "'group' copies videos into person_N/ subfolders. "
        "'rename' copies videos to the output root with person_N_vid_M.ext names. "
        "A video with multiple people is copied once per person in both modes."
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
        "'arcface' uses insightface+RetinaFace (more accurate, 512-D embeddings, "
        "downloads ~300 MB model on first run)."
    ),
)
@click.option(
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="hog",
    show_default=True,
    help=(
        "Face detection model for dlib backend. "
        "'hog' is fast but misses non-frontal or small faces. "
        "'cnn' is much more accurate but significantly slower (GPU recommended). "
        "Ignored when --backend is arcface."
    ),
)
@click.option(
    "--upsample",
    type=int,
    default=1,
    show_default=True,
    help=(
        "How many times to upsample frames before detection (dlib backend only). "
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
        "Minimum number of face detections needed to form a person cluster. "
        "Set to 1 to keep every detected face as its own group."
    ),
)
@click.option(
    "--max-duration",
    type=click.FloatRange(1.0, 120.0),
    default=15.0,
    show_default=True,
    help=(
        "Maximum video duration in seconds. Videos longer than this are skipped "
        "and copied to output/skipped/. Range: 1–120 seconds."
    ),
)
@click.option(
    "--ref-dir",
    "--reference-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory of named reference images. Clusters matching a reference are named "
        "after the image stem: e.g. 'john.jpg' → 'john/'. "
        "Multiple photos per person are supported via a _N suffix: "
        "'john_1.jpg', 'john_2.jpg', ... all map to 'john'."
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
        "Use this to find a good --eps value for your video set."
    ),
)
@click.option(
    "--start-index",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Starting index for file counters in rename mode (e.g. --start-index 10 "
        "produces person_N_vid_10.ext, person_N_vid_11.ext, ...). "
        "Ignored in group mode."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip files already processed in a previous run. Warns if settings differ.",
)
def video_command(
    inputs: tuple[Path, ...],
    output: Path,
    backend: str,
    model: str,
    upsample: int,
    eps: float,
    min_samples: int,
    max_duration: float,
    mode: str,
    ref_dir: Path | None,
    dry_run: bool,
    debug: bool,
    start_index: int,
    resume: bool,
) -> None:
    """Scan INPUTS for videos, detect faces, cluster by person, and organise output.

    INPUTS can be directories (scanned recursively) or individual video files,
    or a mix of both. Supported formats: .mp4 .mov .avi .mkv

    Frames are sampled at 1 fps. Videos with detected faces are grouped into
    person_N/ folders. A video with multiple people is copied into each relevant folder.
    Videos exceeding --max-duration are skipped and copied to output/skipped/.

    Requires opencv-python: pip install 'face-grouper[video]'
    """
    from face_grouper.clusterer import cluster_embeddings, embedding_distance_stats, relabel_by_frequency
    from face_grouper.detector import detect_reference_faces, detect_video_faces
    from face_grouper.video_organizer import organize_videos
    from face_grouper.video_scanner import scan_videos

    # --- 1. Collect inputs ---
    if not inputs:
        raise click.UsageError("No inputs provided. Pass files or directories.")

    console.print(f"[bold]face-grouper[/bold] v{__version__}")

    if max_duration > 30:
        console.print(
            f"[yellow]Warning:[/yellow] --max-duration {max_duration}s is greater than 30s. "
            "Processing may be slow for longer videos."
        )
    console.print(f"Scanning {len(inputs)} input(s) for videos...")

    video_paths = scan_videos(list(inputs))

    # Exclude files inside the output directory to avoid re-processing previous output
    output_resolved = output.resolve()
    pre_exclude = len(video_paths)
    video_paths = [p for p in video_paths if not p.is_relative_to(output_resolved)]
    if len(video_paths) < pre_exclude:
        console.print(
            f"  [dim]Excluded {pre_exclude - len(video_paths)} file(s) already inside "
            f"the output directory (from a previous run).[/dim]"
        )

    # Load previous manifest for --resume
    import datetime
    from face_grouper.run_tracker import (
        RunManifest,
        build_processed_set,
        compare_settings,
        filter_unprocessed,
        load_manifest,
        save_manifest,
    )
    prior_manifest: RunManifest | None = None
    if resume:
        prior_manifest = load_manifest(output)
        if prior_manifest is None:
            console.print("[dim]No previous run found, starting fresh.[/dim]")
        else:
            drift = compare_settings(prior_manifest.settings, {
                "backend": backend, "model": model, "eps": eps,
                "min_samples": min_samples, "mode": mode, "upsample": upsample,
                "max_duration": max_duration, "start_index": start_index,
            })
            for msg in drift:
                console.print(f"  [yellow]Warning:[/yellow] {msg}")
            processed_set = build_processed_set(prior_manifest)
            before = len(video_paths)
            video_paths = filter_unprocessed(video_paths, processed_set)
            skipped_count = before - len(video_paths)
            if skipped_count:
                console.print(f"  [dim]Resuming: skipping {skipped_count} already-processed file(s).[/dim]")
            if not video_paths:
                console.print("[bold green]All inputs already processed.[/bold green]")
                return

    if not video_paths:
        raise click.UsageError("No supported video files found in the provided inputs.")

    console.print(f"Found [cyan]{len(video_paths)}[/cyan] video(s).")

    # --- 2. Overwrite check ---
    if (
        not dry_run
        and not resume
        and output.exists()
        and output.is_dir()
        and any(output.iterdir())
    ):
        click.confirm(
            f"\nOutput directory '{output}' already exists and is not empty. "
            "Existing files will NOT be overwritten — new files will be added with "
            "incremented names if there are conflicts. Continue?",
            abort=True,
        )

    # --- 1b. Load reference embeddings ---
    import re as _re
    reference_embeddings: dict = {}
    if ref_dir is not None:
        from face_grouper.scanner import scan_images
        ref_paths = scan_images([ref_dir])
        if ref_paths:
            potential_names = {(_re.sub(r"_\d+$", "", p.stem) or p.stem) for p in ref_paths}
            reserved_conflicts = {
                name for name in potential_names
                if name == "unknown" or _re.match(r"^person_\d+$", name)
            }
            if reserved_conflicts:
                raise click.UsageError(
                    f"Reference image name(s) conflict with reserved names: "
                    f"{', '.join(sorted(reserved_conflicts))}. "
                    f"Rename them to avoid 'unknown' and 'person_N' patterns."
                )

            console.print(f"Loading reference faces from [cyan]{len(ref_paths)}[/cyan] image(s) in {ref_dir}...")
            reference_embeddings = detect_reference_faces(
                ref_paths, backend=backend, model=model, upsample=upsample
            )

        if not reference_embeddings:
            console.print(
                "[yellow]Warning:[/yellow] No valid reference faces found — "
                "clusters will use default person_N naming."
            )
        else:
            total_photos = sum(len(v) for v in reference_embeddings.values())
            names = ", ".join(f"[bold]{n}[/bold]" for n in sorted(reference_embeddings))
            console.print(
                f"Loaded [cyan]{len(reference_embeddings)}[/cyan] person(s) from "
                f"[cyan]{total_photos}[/cyan] reference photo(s): {names}"
            )

    # --- 2. Detect faces ---
    console.print("\nStep 1/3 — Face detection")
    embeddings, embedded_paths, skipped_paths = detect_video_faces(
        video_paths, model=model, upsample=upsample, backend=backend, max_duration=max_duration
    )

    n_unique = len(set(embedded_paths))
    console.print(
        f"  [green]{len(embeddings)}[/green] face(s) across "
        f"[green]{n_unique}[/green] video(s) with faces, "
        f"[yellow]{len(skipped_paths)}[/yellow] skipped."
    )

    # --- 3. Cluster ---
    console.print("\nStep 2/3 — Clustering")
    labels: list[int] = []
    label_map: dict[int, int] = {}

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

        # Dedup before relabelling: count each (video, label) pair only once so that
        # a video with many frames of the same person doesn't inflate its cluster count.
        seen: set = set()
        deduped: list[int] = []
        for _path, _lbl in zip(embedded_paths, labels):
            key = (_path, _lbl)
            if key not in seen:
                seen.add(key)
                deduped.append(_lbl)
        label_map = relabel_by_frequency(deduped)

        unique_persons = len(label_map)
        name_map = _resolve_reference_names(embeddings, labels, label_map, reference_embeddings, metric, eps)
        outliers = labels.count(-1)
        console.print(
            f"  [green]{unique_persons}[/green] person group(s) found, "
            f"[yellow]{outliers}[/yellow] outlier face(s)."
        )

        if unique_persons == 1 and len(embeddings) > 1:
            suggested = round(max(0.1, eps - 0.1), 2)
            console.print(
                f"  [yellow]Tip:[/yellow] Only 1 group found — try a lower "
                f"[bold]--eps {suggested}[/bold], or run with [bold]--debug[/bold] "
                f"to inspect distance distribution."
            )
        if name_map:
            console.print(
                f"  Reference matches: "
                + ", ".join(f"person_{label_map[lbl]} → [bold]{name}[/bold]" for lbl, name in name_map.items())
            )
    else:
        console.print("  No embeddings to cluster — all videos will go to skipped/.")
        name_map = {}

    # --- 4. Organise ---
    console.print(f"\nStep 3/3 — Organising{'  DRY-RUN' if dry_run else ''}")
    successes, org_errors = organize_videos(
        embedded_paths=embedded_paths,
        skipped_paths=skipped_paths,
        labels=labels,
        label_map=label_map,
        output_dir=output,
        mode=mode,
        dry_run=dry_run,
        name_map=name_map or None,
        start_index=start_index,
    )

    if not dry_run:
        prior_processed = build_processed_set(prior_manifest)
        new_processed = {str(p.resolve()) for p in successes}
        all_processed = sorted(prior_processed | new_processed)
        manifest = RunManifest(
            schema_version=1,
            app_version=__version__,
            command="video",
            run_date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            inputs=[str(Path(p).resolve()) for p in inputs],
            settings={
                "backend": backend, "model": model, "eps": eps,
                "min_samples": min_samples, "mode": mode, "upsample": upsample,
                "max_duration": max_duration, "start_index": start_index,
            },
            stats={
                "n_videos_scanned": len(video_paths),
                "n_faces_detected": len(embeddings),
                "n_persons_found": len(label_map),
                "n_skipped": len(skipped_paths),
            },
            processed_files=all_processed,
        )
        save_manifest(output, manifest)
    if org_errors:
        raise SystemExit(1)
