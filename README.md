# face-organizer

A CLI tool that groups photos by person using offline face recognition.

## Installation

```bash
# Default (dlib backend)
pipx install face-grouper

# With ArcFace support (more accurate, recommended)
pipx install "face-grouper[arcface]"
```

> **Compile error?** Install CMake first: `brew install cmake` (macOS), `sudo apt install cmake build-essential` (Linux), or [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Windows).

Don't have pipx? `brew install pipx && pipx ensurepath` (macOS) or `pip install --user pipx && pipx ensurepath` (Linux/Windows).

## Usage

```bash
# Group into person_1/, person_2/, ... subfolders (default)
fgroup group ./photos --output ./sorted

# Flat output with renamed files: person_1_img_1.jpg, person_2_img_1.jpg, ...
fgroup group ./photos --output ./sorted --mode rename

# ArcFace backend — more accurate when dlib groups everyone into 1 person
# (~300 MB model downloaded to ~/.insightface/ on first run, then cached offline)
fgroup group ./photos --output ./sorted --backend arcface

# Include people who appear in only 1 photo (default min-samples=2 sends them to unknown/)
fgroup group ./photos --output ./sorted --backend arcface --min-samples 1

# Too many groups (same person split)? raise --eps. Too few? lower it.
fgroup group ./photos --output ./sorted --backend arcface --eps 0.6

# Not sure what --eps to use? --debug prints distance distribution to guide you
fgroup group ./photos --output ./sorted --backend arcface --debug

# Preview without copying anything
fgroup group ./photos --output ./sorted --dry-run

# Pass individual files instead of a folder
fgroup group a.jpg b.jpg c.jpg --output ./sorted

# dlib only: better detection for small/angled faces (slower)
fgroup group ./photos --output ./sorted --model cnn --upsample 2
```

Originals are **never** modified. Supported formats: `.jpg` `.jpeg` `.png` `.webp` `.bmp`

Full option reference: `fgroup group --help`

## Dev mode

```bash
git clone https://github.com/your-username/face-organizer.git
cd face-organizer
pip install -e .
```

Changes to `face_grouper/*.py` take effect immediately — no reinstall needed. Only re-run `pip install -e .` if you change `pyproject.toml` (e.g. new dependency or entry point).
