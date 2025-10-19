# Changelog

## [Unreleased]

## [1.1.0] - 2025-10-19

### Added

- Python 3.13 and 3.13t (free-threaded) support with PYTHON_GIL=0 configuration
- Pre-commit configuration with Ruff, mdformat, and file validators
- New GitHub Actions workflows for pre-commit checks and security scanning
- Packaging compatibility workflows for cx_Freeze, PyInstaller, and py2exe
- Standalone test scripts and demo builders for verifying packaged distributions
- Coverage HTML artifact uploads to all test workflows
- Concurrency controls to workflows to cancel redundant builds
- Justfile for streamlined development tasks (setup, docs, test, typecheck, lint)
- Security scanning with pip-audit and bandit

### Changed

- Minimum Python version raised from 3.6 to 3.8
- Updated actions to latest versions (setup-qemu v3, run-on-arch v3)
- Updated macOS CI runners (macos-13 → macos-15-intel, added macos-latest for ARM)
- Extensive code quality improvements with Ruff linting across entire codebase:
  - String quote normalization to double quotes
  - Removed unused imports and variables
  - Improved code formatting and PEP 8 compliance
  - Better type hints compatibility
- Merged type hints across the codebase with improved accuracy
- Updated copyright year to 2025
- Fixed numerous typos across documentation files
- Improved Sphinx configuration for Python 3.13 compatibility
- Enhanced docstring and markdown formatting
- Updated README to reflect Python 3.8+ requirement

### Removed

- Dropped Python 3.6 and 3.7 support
- Removed obsolete platform-specific exclusions

### Security

- Added automated security scanning workflow documented in SECURITY.md

## [1.0.7] - 2024-11-10

### Changed

- Code cleanup and minor optimizations
- Fix Issue #54 https://github.com/CalebBell/fluids/issues/54
- Fluids version dependency now >= 1.0.27

## [1.0.6] - 2024-07-26

### Changed

- Compatibility with NumPy 2.0 and SciPy 1.14
- Fluids version dependency now >= 1.0.26

## [1.0.5] - 2023-06-04

### Changed

- Code cleanup with ruff (experiment)

## [1.0.4] - 2023-04-23

### Added

- Nothing

### Changed

- Internal cleanup
- Fix to Nu_plate_Martin correlation (see https://github.com/CalebBell/ht/pull/8)

### Removed

- Support for Python before 3.6
- Drop appveyor and Travis CI

### Fixed

- Nothing
