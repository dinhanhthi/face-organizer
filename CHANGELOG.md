# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
