# Contributing

Thank you for contributing to `forestgeom`!

## Clone the repository

```bash
git clone https://github.com/JakeSRhodesLab/ForestGeom.git
cd ForestGeom
```

## Create a development environment

We recommend [`uv`](https://docs.astral.sh/uv/) for development. From the
repository root, create a local environment and install the project with test
dependencies:

```bash
uv sync --extra test
```

Install optional dependency groups only when needed:

```bash
# boosted tree support
uv sync --extra boosted --extra test

# visualization dependencies
uv sync --extra viz --extra test

# experiment dependencies
uv sync --extra experiments --extra test
```

`uv sync` installs the project in editable mode, so local source changes are
immediately importable.

<details>
<summary>pip also works</summary>

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
python -m pip install -e ".[boosted]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[experiments]"
```

</details>

## Build distributions

The project uses Hatchling as its PEP 517 build backend. To verify that the
source distribution and wheel build correctly, run:

```bash
uv build
```

Build artifacts are written to `dist/`.

If you are preparing a release, clean old artifacts before building:

```bash
rm -rf dist/
uv build
```

## Release publishing

Releases are tag-driven. The GitHub deploy workflow in
`.github/workflows/deploy.yml` runs only when a tag is pushed, and it uses the
tag name to decide whether to publish to TestPyPI or PyPI.

Release tag conventions:

- `test-v<version>` publishes to TestPyPI and creates a GitHub prerelease.
- `v<version>` publishes to PyPI and creates a normal GitHub release.

The workflow checks that the pushed tag matches `__version__` in
`src/forestgeom/version.py` before publishing.

GitHub environment protection is enabled for both `testpypi` and `pypi`, so
selected approved maintainers must review the deploy before publication.

Recommended release steps:

1. Bump `__version__` in `src/forestgeom/version.py`.
2. Build and test locally.
3. Create the release tag that matches the version:

```bash
git tag -a test-v0.0.4 -m "Release v0.0.4 to TestPyPI"
git push origin test-v0.0.4

# or, for the final PyPI release
git tag -a v0.0.4 -m "Release v0.0.4"
git push origin v0.0.4
```

4. Approve the deploy in GitHub if your environment requires review.

The workflow will then build the package, publish to the correct index, and
attach the release artifacts to GitHub.

## Run tests

The project uses three levels of testing. Use the lowest level that answers the
question you have, and run broader checks before releases or dependency changes.

### 1. Current development environment

For ordinary local development, run pytest in your active `uv` environment:

```bash
uv run pytest
```

This is the fastest feedback loop. It tests the code against the exact Python
interpreter and dependency versions installed in your local `.venv`.

To include optional boosted-tree adapters in this local environment, sync the
extra first:

```bash
uv sync --extra boosted --extra test
uv run pytest
```

### 2. Local compatibility sweep

Use `tox` when you need to check package metadata against multiple Python and
dependency scenarios on your machine. The tox environments are defined in
`tox.ini`, and dependency lower-bound scenarios are pinned with files in
`constraints/`.

Run one scenario:

```bash
uvx tox -e py310-lowest-core
uvx tox -e py310-latest-boosted
```

Run the full local sweep:

```bash
uvx tox
```

Run the full local sweep in parallel with live output:

```bash
uvx tox -p auto --parallel-live
```

The sweep covers:

- lowest supported core dependency stack on Python 3.10
- latest core dependency stack across supported Python versions
- scikit-learn compatibility scenarios from 1.3 through 1.9
- lowest and latest optional boosted dependency stacks

If a scenario is skipped, the corresponding Python interpreter is probably not
installed locally. GitHub Actions installs the requested Python versions
explicitly.

### 3. GitHub Actions matrix

The highest-level check is the GitHub Actions test workflow in
`.github/workflows/test.yml`. It runs the tox compatibility matrix on CI: Linux
handles the full Python/dependency sweep, while macOS and Windows run selected
smoke scenarios to catch machine-specific issues without duplicating the entire
matrix.

Use the GitHub workflow as the final source of truth before publishing or
merging dependency-bound changes.

## Notes

- Use a virtual environment for development work.
- `uv sync` is recommended for contributors; pip editable installs remain
  supported.
- Use `uv run pytest` for day-to-day checks, `uvx tox` for local
  compatibility sweeps, and the GitHub Actions workflow for the full remote
  matrix.
- Hatchling is configured in `pyproject.toml`; contributors do not need a
  `setup.py` or `setup.cfg`.
- Some optional dependencies (e.g. `lightgbm` or `xgboost`) may require
  additional system-level installation steps. Refer to the corresponding
  project documentation if needed.
