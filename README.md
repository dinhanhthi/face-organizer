<p align="center">
  <img src="logo.png" alt="face-organizer Logo" width="100" />
</p>

<h1 align="center">face-organizer</h1>

<p align="center">
  A CLI tool that groups photos by person using offline face recognition.
</p>

## Installation

```bash
# Default (dlib backend)
pipx install face-grouper

# With ArcFace support (more accurate, recommended)
pipx install "face-grouper[arcface]"

# Upgrade (core only)
pipx upgrade face-grouper

# Upgrade with ArcFace support
pipx install "face-grouper[arcface]" --force
```

> **Compile error?** Install CMake first: `brew install cmake` (macOS), `sudo apt install cmake build-essential` (Linux), or [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Windows).

Don't have pipx? `brew install pipx && pipx ensurepath` (macOS) or `pip install --user pipx && pipx ensurepath` (Linux/Windows).

## Usage

```bash
# ⭐ Quick start — named clusters, flat output, include single-photo people
fgroup group ./photos --reference-dir ./refs --output ./sorted --backend arcface --mode rename --min-samples 1

# Basic grouping into person_1/, person_2/, ... subfolders
fgroup group ./photos --output ./sorted

# Pass individual files
fgroup group a.jpg b.jpg c.jpg --output ./sorted

# Disable multi-face export (legacy: one output per photo)
fgroup group ./photos --output ./sorted --no-multi-export
```

Originals are **never** modified. Supported formats: `.jpg` `.jpeg` `.png` `.webp` `.bmp`

## Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `INPUTS...` | _(required)_ | Directories or image files to scan (recursive) |
| `--output / -o` | _(required)_ | Output directory |
| `--backend` | `dlib` | `dlib` (fast, 128-D) or `arcface` (accurate, 512-D, downloads ~300 MB on first run) |
| `--mode` | `group` | `group` → `person_N/` subfolders; `rename` → flat `person_N_img_M.ext` files |
| `--keep-originals` | `false` | Accepted for compatibility — originals are never modified in any mode |
| `--eps` | `0.5` | DBSCAN max distance between embeddings for same person. Raise to merge split clusters, lower to split merged ones. Use `--debug` to calibrate |
| `--min-samples` | `2` | Min photos to form a cluster. People below threshold go to `unknown/`. Set to `1` to keep solo faces |
| `--reference-dir` | _none_ | Folder of named reference images. `john.jpg` → cluster named `john`. Multiple photos per person supported: `john_1.jpg`, `john_2.jpg`, … all map to `john` |
| `--no-multi-export` | `false` | Only use the largest face per photo. By default every detected face is exported independently — a photo with two people lands in both person folders. |
| `--model` | `hog` | dlib only: `hog` (fast) or `cnn` (accurate, GPU recommended) |
| `--upsample` | `1` | dlib only: upsample N times before detection — finds smaller faces, ~4× cost per level |
| `--dry-run` | `false` | Preview planned operations without copying anything |
| `--debug` | `false` | Print pairwise distance stats to help choose `--eps` |

## Dev mode

```bash
git clone https://github.com/your-username/face-organizer.git
cd face-organizer
pip install -e .
```

Changes to `face_grouper/*.py` take effect immediately — no reinstall needed. Only re-run `pip install -e .` if you change `pyproject.toml` (e.g. new dependency or entry point).
