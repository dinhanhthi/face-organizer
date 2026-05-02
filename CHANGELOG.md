# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-05-01

### Added
- `--start-index` option for both `fgroup group` and `fgroup video` in rename mode: sets the starting counter index (e.g. `--start-index 10` produces `person_N_img_10.ext`, `person_N_img_11.ext`, …); defaults to `1`, ignored in group mode ([49d27ef](https://github.com/dinhanhthi/face-grouper/commit/49d27ef))

### Changed
- README Parameters section restructured into a shared common table and per-command specific tables ([49d27ef](https://github.com/dinhanhthi/face-grouper/commit/49d27ef))

## [0.3.0] - 2026-05-01

### Added
- `fgroup video` subcommand: groups video files by detected person using the same face recognition pipeline as `fgroup group` ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))
- `--max-duration` option for `fgroup video`: default 15s, max 120s — videos exceeding the limit go to `skipped/` ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))
- `--ref-dir` support for `fgroup video`: maps reference images to named clusters (e.g. `john.jpg` → `john/` folder) ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))
- `--mode [group|rename]` for `fgroup video`: `group` creates `person_N/` subfolders, `rename` produces flat `person_N_vid_M.ext` files ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))
- Multi-person video support: a video featuring N people is copied into all N person folders ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))
- New `[video]` optional extra: `pip install 'face-grouper[video]'` pulls in `opencv-python` ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))

### Changed
- `--reference-dir` renamed to `--ref-dir` in `fgroup group` (old name kept as alias for compatibility) ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))
- `collision_free_path()` made public API (was `_collision_free_path`) ([66f6967](https://github.com/dinhanhthi/face-grouper/commit/66f6967))

## [0.2.0] - 2026-05-01

### Added
- Multi-face-per-photo export: photos with N faces produce N copies in output by default; use `--no-multi-export` to revert to one copy per photo ([4f1cabf](https://github.com/dinhanhthi/face-grouper/commit/4f1cabf))
- `--reference-dir` now accepts multiple photos per person for more robust identity matching ([54d775a](https://github.com/dinhanhthi/face-grouper/commit/54d775a))

### Changed
- Added project logo to README ([26b0dd8](https://github.com/dinhanhthi/face-grouper/commit/26b0dd8))

## [0.1.1] - 2026-05-01

### Fixed
- GitHub Actions publish workflow now skips already-uploaded files to PyPI instead of failing (`skip_existing: true`)
- GitHub Actions automatically creates a GitHub Release with CHANGELOG notes on tag push

## [0.1.0] - 2026-05-01

### Added
- `fgroup group` CLI command with dlib and ArcFace backends
- DBSCAN-based face clustering with configurable `--eps` and `--min-samples`
- `--mode` flag: `folder` (default) or `rename` for flat output
- `--reference-dir` to name clusters after known people
- `--dry-run` to preview without copying files
- `--debug` to print distance distribution for tuning `--eps`
- CNN model support for dlib (`--model cnn --upsample`)
- GitHub Actions workflow to auto-publish to PyPI on version tags
