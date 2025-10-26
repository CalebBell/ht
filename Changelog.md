# Changelog

## [Unreleased]

## [1.2.0] - 2025-10-26

### Added

- Project is now PEP 517 compliant and doesn't use deprecated setup.py commands anymore
- GitHub Actions workflow for publishing to PyPI using environment protection
- `uv` package manager integration across CI workflows for faster dependency resolution
- New consolidated `quality.yml` workflow for linting and testing
- New `build_third_party_packagers.yml` workflow consolidating cx_Freeze, PyInstaller, Nuitka, and py2exe testing
- Pre-commit hooks configuration
- Justfile with extensive development automation commands - github actions have been refactored to use this where possible, making them locally debuggable

### Changed

- **Breaking**: Dropped Python 3.8 support; minimum Python version is now 3.9
- Migrated from setup.py to pyproject.toml-only configuration (PEP 517)
- Consolidated multiarch CI workflows with reduced test matrix for improved performance
- Optimized multi-numpy/scipy testing workflow using `uv`
- Simplified third-party packager testing into single consolidated workflow
- Moved coverage configuration to pyproject.toml
- Moved pytest configuration from pytest.ini to pyproject.toml
- Moved mypy configuration from mypy.ini to pyproject.toml
- Moved Ruff configuration from .ruff.toml to pyproject.toml
- Reorganized development requirements into pyproject.toml optional dependencies
- Updated fluids dependency to >= 1.2.0

### Removed

- Removed setup.py (replaced by pyproject.toml)
- Removed standalone configuration files: pytest.ini, mypy.ini, .ruff.toml
- Removed separate workflow files: build_cxfreeze_library.yml, build_nuitka_library.yml, build_py2exe_library.yml, build_pyinstaller_library.yml
- Removed pre-commit.yml and security.yml workflows (consolidated into quality.yml)
- Removed separate requirements files (test, docs, multiarch) - now in pyproject.toml

### Security

- Implemented PyPI publishing workflow with manual approval gate

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
