# Contributing

Thank you for contributing to `forestgeom`.

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
python -m pip install -U pip build twine
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

If you are preparing a release, validate the built artifacts with Twine:

```bash
uv run twine check dist/*
```

Uploading to PyPI is also done with Twine, but only by maintainers during a
release:

```bash
uv run twine upload dist/*
```

## Run tests

```bash
uv run pytest
```

## Notes

- Use a virtual environment for development work.
- `uv sync` is recommended for contributors; pip editable installs remain
  supported.
- Hatchling is configured in `pyproject.toml`; contributors do not need a
  `setup.py` or `setup.cfg`.
- Some optional dependencies (e.g. `lightgbm` or `xgboost`) may require
  additional system-level installation steps. Refer to the corresponding
  project documentation if needed.
