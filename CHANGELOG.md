# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-01

### Added
- Multi-face-per-photo export: photos with N faces produce N copies in output by default; use `--no-multi-export` to revert to one copy per photo ([4f1cabf](https://github.com/dinhanhthi/face-organizer/commit/4f1cabf))
- `--reference-dir` now accepts multiple photos per person for more robust identity matching ([54d775a](https://github.com/dinhanhthi/face-organizer/commit/54d775a))

### Changed
- Added project logo to README ([26b0dd8](https://github.com/dinhanhthi/face-organizer/commit/26b0dd8))

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
