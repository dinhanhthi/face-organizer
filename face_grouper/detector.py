"""Face detection and embedding extraction using face_recognition or DeepFace/ArcFace."""

import re
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def _face_area(loc: tuple[int, int, int, int]) -> int:
    """Compute bounding-box area from a face_locations tuple (top, right, bottom, left)."""
    top, right, bottom, left = loc
    return (right - left) * (bottom - top)


def _detect_dlib(
    image_paths: list[Path],
    model: str = "hog",
    upsample: int = 1,
    multi_face: bool = True,
) -> tuple[list[np.ndarray], list[Path], list[Path]]:
    """Detect faces using dlib/face_recognition and extract 128-D embeddings.

    When multi_face is False and multiple faces are found, picks the one with the
    largest bounding box. When multi_face is True, extracts all detected faces,
    yielding one (embedding, path) pair per face. Images with no detected faces
    are collected separately.

    Args:
        image_paths: List of image file paths to process.
        model: Detection model — "hog" (fast) or "cnn" (accurate, GPU recommended).
        upsample: Number of times to upsample before detection; higher finds smaller faces.
        multi_face: When True, extract all faces per image (one embedding per face).
            When False, extract only the largest face per image.

    Returns:
        A tuple of:
        - embeddings: List of 128-D numpy arrays, one per detected face.
        - embedded_paths: Paths corresponding to each embedding (may contain duplicates
            when multi_face is True and an image has multiple faces).
        - no_face_paths: Paths of images where no face was detected.
    """
    import face_recognition  # noqa: PLC0415 — deferred to avoid import cost

    embeddings: list[np.ndarray] = []
    embedded_paths: list[Path] = []
    no_face_paths: list[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Detecting faces...", total=len(image_paths))

        for image_path in image_paths:
            display = f"{image_path.parent.name}/{image_path.name}"
            progress.update(task, description=f"Processing {display}")

            image = face_recognition.load_image_file(str(image_path))
            # face_locations returns (top, right, bottom, left) tuples
            face_locations = face_recognition.face_locations(
                image, number_of_times_to_upsample=upsample, model=model
            )

            if not face_locations:
                no_face_paths.append(image_path)
                progress.advance(task)
                continue

            if multi_face:
                # Encode all detected faces; zip is length-safe (face_recognition
                # can return fewer encodings than locations in rare edge cases).
                encodings = face_recognition.face_encodings(
                    image, known_face_locations=face_locations
                )
                added = 0
                for _loc, encoding in zip(face_locations, encodings):
                    embeddings.append(encoding)
                    embedded_paths.append(image_path)
                    added += 1
                if added == 0:
                    # Detection found faces but encoding failed for all of them.
                    no_face_paths.append(image_path)
            else:
                if len(face_locations) == 1:
                    largest_location = face_locations[0]
                else:
                    # Pick face with largest bounding-box area
                    largest_location = max(face_locations, key=_face_area)

                # Encode only the selected face for efficiency
                encodings = face_recognition.face_encodings(
                    image, known_face_locations=[largest_location]
                )

                if encodings:
                    embeddings.append(encodings[0])
                    embedded_paths.append(image_path)
                else:
                    no_face_paths.append(image_path)

            progress.advance(task)

    return embeddings, embedded_paths, no_face_paths


def _detect_arcface(
    image_paths: list[Path],
    multi_face: bool = True,
) -> tuple[list[np.ndarray], list[Path], list[Path]]:
    """Detect faces using insightface (RetinaFace+ArcFace) and extract 512-D embeddings.

    Uses ONNX runtime — no TensorFlow required. Downloads ~300 MB buffalo_l model
    to ~/.insightface/ on first run.

    When multi_face is False and multiple faces are found, picks the one with the
    largest bounding box area. When multi_face is True, extracts all detected faces,
    yielding one (embedding, path) pair per face. Images with no detected faces
    are collected separately.

    Args:
        image_paths: List of image file paths to process.
        multi_face: When True, extract all faces per image (one embedding per face).
            When False, extract only the largest face per image.

    Returns:
        A tuple of:
        - embeddings: List of 512-D numpy arrays, one per detected face.
        - embedded_paths: Paths corresponding to each embedding (may contain duplicates
            when multi_face is True and an image has multiple faces).
        - no_face_paths: Paths of images where no face was detected.
    """
    import cv2  # noqa: PLC0415
    from insightface.app import FaceAnalysis  # noqa: PLC0415 — deferred, optional dep

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings: list[np.ndarray] = []
    embedded_paths: list[Path] = []
    no_face_paths: list[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Detecting faces (ArcFace)...", total=len(image_paths))

        for image_path in image_paths:
            display = f"{image_path.parent.name}/{image_path.name}"
            progress.update(task, description=f"Processing {display}")

            img = cv2.imread(str(image_path))
            if img is None:
                no_face_paths.append(image_path)
                progress.advance(task)
                continue

            faces = app.get(img)
            if not faces:
                no_face_paths.append(image_path)
            elif multi_face:
                # Each Face object already has .embedding computed by app.get().
                for face in faces:
                    embeddings.append(face.embedding)
                    embedded_paths.append(image_path)
            else:
                # bbox is [x1, y1, x2, y2]
                largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(largest.embedding)
                embedded_paths.append(image_path)

            progress.advance(task)

    return embeddings, embedded_paths, no_face_paths


def detect_faces(
    image_paths: list[Path],
    model: str = "hog",
    upsample: int = 1,
    backend: str = "dlib",
    multi_face: bool = True,
) -> tuple[list[np.ndarray], list[Path], list[Path]]:
    """Detect faces in images and extract embeddings.

    Dispatches to the appropriate backend:
    - "dlib": uses face_recognition library, produces 128-D embeddings.
    - "arcface": uses DeepFace+RetinaFace+ArcFace, produces 512-D embeddings.
      Requires the optional 'arcface' extra: pip install 'face-grouper[arcface]'.
      Downloads ~300 MB model to ~/.insightface/ on first run.

    When multi_face is False and multiple faces are found in an image, picks the
    one with the largest bounding box. When multi_face is True (default), all
    detected faces are extracted — embedded_paths may contain the same path
    multiple times, once per face found in that image.

    Args:
        image_paths: List of image file paths to process.
        model: Detection model for dlib backend — "hog" (fast) or "cnn" (accurate,
            GPU recommended). Ignored when backend is "arcface".
        upsample: Number of times to upsample before detection for dlib backend;
            higher finds smaller faces. Ignored when backend is "arcface".
        backend: Face recognition backend — "dlib" (default, 128-D) or "arcface"
            (more accurate, 512-D).
        multi_face: When True (default), extract all faces per image, yielding one
            embedding per detected face. When False, extract only the largest face
            per image (original behaviour).

    Returns:
        A tuple of:
        - embeddings: List of numpy arrays (128-D for dlib, 512-D for arcface),
            one per detected face.
        - embedded_paths: Paths corresponding to each embedding. When multi_face
            is True, the same path may appear multiple times (once per face).
        - no_face_paths: Paths of images where no face was detected.
    """
    if backend.lower() == "arcface":
        return _detect_arcface(image_paths, multi_face=multi_face)
    return _detect_dlib(image_paths, model=model, upsample=upsample, multi_face=multi_face)


def _reference_base_name(stem: str) -> str:
    """Extract person name from a reference filename stem by stripping trailing _N suffix.

    'john_1' → 'john', 'john_doe_2' → 'john_doe', 'john' → 'john'.
    Falls back to the original stem if stripping would produce an empty string.
    """
    return re.sub(r"_\d+$", "", stem) or stem


def detect_reference_faces(
    image_paths: list[Path],
    backend: str = "dlib",
    model: str = "hog",
    upsample: int = 1,
) -> dict[str, list[np.ndarray]]:
    """Detect one face per reference image and return a name → embeddings mapping.

    Filenames with a trailing _N suffix (e.g. john_1.jpg, john_2.jpg) are grouped
    under the same base name ("john"), allowing multiple reference photos per person.
    Images with zero or more than one face are skipped with a warning.
    Must be called with the same backend/model/upsample settings as the main detection run
    so that embedding spaces are compatible.

    Args:
        image_paths: Reference image paths (typically from a --reference-dir scan).
        backend: "dlib" or "arcface" — must match the backend used for main detection.
        model: dlib detection model ("hog" or "cnn"). Ignored for arcface.
        upsample: Upsample factor for dlib. Ignored for arcface.

    Returns:
        Mapping from person name to list of embedding arrays (one per reference photo).
        Multiple photos for the same person are accumulated under the same key.
    """
    result: dict[str, list[np.ndarray]] = {}

    if backend.lower() == "arcface":
        import cv2  # noqa: PLC0415
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                console.print(f"[yellow]Warning:[/yellow] {path.name} — could not load image, skipping.")
                continue
            faces = app.get(img)
            if len(faces) == 0:
                console.print(f"[yellow]Warning:[/yellow] {path.name} — no face detected, skipping.")
            elif len(faces) > 1:
                console.print(
                    f"[yellow]Warning:[/yellow] {path.name} — {len(faces)} faces detected, skipping "
                    f"(reference images must contain exactly one face)."
                )
            else:
                base = _reference_base_name(path.stem)
                if base != path.stem:
                    console.print(f"  [dim]{path.name} → grouped under '{base}'[/dim]")
                result.setdefault(base, []).append(faces[0].embedding)
    else:
        import face_recognition  # noqa: PLC0415

        for path in image_paths:
            image = face_recognition.load_image_file(str(path))
            face_locations = face_recognition.face_locations(
                image, number_of_times_to_upsample=upsample, model=model
            )
            if len(face_locations) == 0:
                console.print(f"[yellow]Warning:[/yellow] {path.name} — no face detected, skipping.")
            elif len(face_locations) > 1:
                console.print(
                    f"[yellow]Warning:[/yellow] {path.name} — {len(face_locations)} faces detected, skipping "
                    f"(reference images must contain exactly one face)."
                )
            else:
                encodings = face_recognition.face_encodings(
                    image, known_face_locations=[face_locations[0]]
                )
                if encodings:
                    base = _reference_base_name(path.stem)
                    if base != path.stem:
                        console.print(f"  [dim]{path.name} → grouped under '{base}'[/dim]")
                    result.setdefault(base, []).append(encodings[0])
                else:
                    console.print(f"[yellow]Warning:[/yellow] {path.name} — encoding failed, skipping.")

    return result
