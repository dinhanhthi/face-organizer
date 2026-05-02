<p align="center">
  <img src="logo.png" alt="face-grouper Logo" width="100" />
</p>

<h1 align="center">face-grouper</h1>

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

### Common parameters (`fgroup group` and `fgroup video`)

| Option | Default | Description |
|--------|---------|-------------|
| `INPUTS...` | _(required)_ | Directories or files to scan (recursive) |
| `--output / -o` | _(required)_ | Output directory |
| `--backend` | `dlib` | Face recognition backend. `dlib`: fast, 128-D embeddings, works offline, good for frontal faces — pair with `--model cnn` for better accuracy. `arcface`: 512-D embeddings, more accurate across varied lighting/angles, but downloads around 300 MB (`buffalo_l` model) on first run to `~/.insightface/`; `--model` and `--upsample` are ignored for this backend |
| `--mode` | `group` | `group` → `person_N/` subfolders; `rename` → flat files with sequential index |
| `--start-index` | `1` | Starting index for file counters in rename mode (e.g. `--start-index 10` → `person_N_img_10.ext`, `person_N_img_11.ext`, …). Ignored in group mode |
| `--eps` | `0.5` | DBSCAN max distance between embeddings for same person. Raise to merge split clusters, lower to split merged ones. Use `--debug` to calibrate |
| `--min-samples` | `2` | Min detections to form a cluster. People below threshold go to `unknown/`. Set to `1` to keep solo faces |
| `--ref-dir` | _none_ | Folder of named reference images. `john.jpg` → cluster named `john`. Multiple photos per person supported: `john_1.jpg`, `john_2.jpg`, … all map to `john` |
| `--model` | `hog` | dlib only: `hog` (fast) or `cnn` (accurate, GPU recommended) |
| `--upsample` | `1` | dlib only: upsample N times before detection — finds smaller faces, ~4× cost per level |
| `--dry-run` | `false` | Preview planned operations without copying anything |
| `--debug` | `false` | Print pairwise distance stats to help choose `--eps` |

### `fgroup group` specific

| Option | Default | Description |
|--------|---------|-------------|
| `--no-multi-export` | `false` | Only use the largest face per photo. By default every detected face is exported independently — a photo with two people lands in both person folders |

### `fgroup video` specific

| Option | Default | Description |
|--------|---------|-------------|
| `--max-duration` | `15.0` | Max video length in seconds (1–120). Videos over this go to `skipped/`. Warns if > 30s |

## Output directory behaviour

If the output directory already exists and is not empty, `fgroup` will ask for confirmation before proceeding. Existing files are **never overwritten** — if a new file would collide with an existing filename, the new file is written with an incremented name (e.g. `img.jpg` → `img_1.jpg` → `img_2.jpg`). This makes it safe to run `fgroup` on an existing output folder; previous results are preserved.

## Dev mode

```bash
git clone https://github.com/your-username/face-grouper.git
cd face-grouper

# Install with all optional extras for full local development
pip install -e ".[arcface,video]"

# Verify the CLI is wired up
fgroup --help
```

Changes to `face_grouper/*.py` take effect immediately — no reinstall needed. Only re-run `pip install -e .` if you change `pyproject.toml` (e.g. new dependency or entry point).

> **macOS compile error?** Install CMake first: `brew install cmake`
