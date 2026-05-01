<p align="center">
  <img src="logo.png" alt="face-organizer Logo" width="100" />
</p>

<h1 align="center">face-organizer</h1>

<p align="center">
  A CLI tool that groups photos and videos by person using offline face recognition.
</p>

## Installation

```bash
# Default (dlib backend, photos only)
pipx install face-grouper

# With video support
pipx install "face-grouper[video]"

# With ArcFace support (more accurate, recommended)
pipx install "face-grouper[arcface]"

# With both video + ArcFace
pipx install "face-grouper[video,arcface]"

# Upgrade (core only)
pipx upgrade face-grouper
```

> **Compile error?** Install CMake first: `brew install cmake` (macOS), `sudo apt install cmake build-essential` (Linux), or [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Windows).

Don't have pipx? `brew install pipx && pipx ensurepath` (macOS) or `pip install --user pipx && pipx ensurepath` (Linux/Windows).

## Usage

### Photos (`fgroup group`)

```bash
# ⭐ Quick start — named clusters, flat output, include single-photo people
fgroup group ./photos --ref-dir ./refs --output ./sorted --backend arcface --mode rename --min-samples 1

# Basic grouping into person_1/, person_2/, ... subfolders
fgroup group ./photos --output ./sorted

# Pass individual files
fgroup group a.jpg b.jpg c.jpg --output ./sorted

# Disable multi-face export (legacy: one output per photo)
fgroup group ./photos --output ./sorted --no-multi-export
```

Originals are **never** modified. Supported formats: `.jpg` `.jpeg` `.png` `.webp` `.bmp`

### Videos (`fgroup video`)

```bash
# Group short clips by person
fgroup video ./clips --output ./sorted

# Allow longer videos (default 15s, max 120s)
fgroup video ./clips --output ./sorted --max-duration 30

# Use ArcFace for better accuracy
fgroup video ./clips --output ./sorted --backend arcface

# Preview without copying
fgroup video ./clips --output ./sorted --dry-run
```

Supported formats: `.mp4` `.mov` `.avi` `.mkv`. Requires `pip install 'face-grouper[video]'`.

Videos exceeding `--max-duration` are skipped and copied to `output/skipped/`. A video with multiple people is copied into each relevant person folder.

## Parameters

### `fgroup group`

| Option | Default | Description |
|--------|---------|-------------|
| `INPUTS...` | _(required)_ | Directories or image files to scan (recursive) |
| `--output / -o` | _(required)_ | Output directory |
| `--backend` | `dlib` | Face recognition backend. `dlib`: fast, 128-D embeddings, works offline, good for frontal faces — pair with `--model cnn` for better accuracy. `arcface`: 512-D embeddings, more accurate across varied lighting/angles, but downloads around 300 MB (`buffalo_l` model) on first run to `~/.insightface/`; `--model` and `--upsample` are ignored for this backend |
| `--mode` | `group` | `group` → `person_N/` subfolders; `rename` → flat `person_N_img_M.ext` files |
| `--eps` | `0.5` | DBSCAN max distance between embeddings for same person. Raise to merge split clusters, lower to split merged ones. Use `--debug` to calibrate |
| `--min-samples` | `2` | Min photos to form a cluster. People below threshold go to `unknown/`. Set to `1` to keep solo faces |
| `--ref-dir` | _none_ | Folder of named reference images. `john.jpg` → cluster named `john`. Multiple photos per person supported: `john_1.jpg`, `john_2.jpg`, … all map to `john` |
| `--no-multi-export` | `false` | Only use the largest face per photo. By default every detected face is exported independently — a photo with two people lands in both person folders. |
| `--model` | `hog` | dlib only: `hog` (fast) or `cnn` (accurate, GPU recommended) |
| `--upsample` | `1` | dlib only: upsample N times before detection — finds smaller faces, ~4× cost per level |
| `--dry-run` | `false` | Preview planned operations without copying anything |
| `--debug` | `false` | Print pairwise distance stats to help choose `--eps` |

### `fgroup video`

| Option | Default | Description |
|--------|---------|-------------|
| `INPUTS...` | _(required)_ | Directories or video files to scan (recursive) |
| `--output / -o` | _(required)_ | Output directory |
| `--mode` | `group` | `group` → `person_N/` subfolders; `rename` → flat `person_N_vid_M.ext` files |
| `--backend` | `dlib` | `dlib` or `arcface` — same as `fgroup group` |
| `--model` | `hog` | dlib only: `hog` or `cnn` |
| `--upsample` | `1` | dlib only: upsample N times before detection |
| `--eps` | `0.5` | DBSCAN epsilon — same as `fgroup group` |
| `--min-samples` | `2` | Min face detections to form a cluster |
| `--ref-dir` | _none_ | Folder of named reference images — same behaviour as `fgroup group` |
| `--max-duration` | `15.0` | Max video length in seconds (1–120). Videos over this go to `skipped/`. Warns if > 30s |
| `--dry-run` | `false` | Preview planned operations without copying anything |
| `--debug` | `false` | Print pairwise distance stats to help choose `--eps` |

## Dev mode

```bash
git clone https://github.com/your-username/face-organizer.git
cd face-organizer

# Install with all optional extras for full local development
pip install -e ".[arcface,video]"

# Verify the CLI is wired up
fgroup --help
```

Changes to `face_grouper/*.py` take effect immediately — no reinstall needed. Only re-run `pip install -e .` if you change `pyproject.toml` (e.g. new dependency or entry point).

> **macOS compile error?** Install CMake first: `brew install cmake`
