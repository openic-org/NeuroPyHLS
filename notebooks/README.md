# Notebooks

Notebooks are stored as Jupytext **percent-format `.py` files**, in which `# %%` marks a cell. This keeps diffs readable and avoids committing outputs. The `.ipynb` copies are git-ignored.

## In VS Code (default editor)

VS Code is the project's default editor. Open the repository folder and accept the recommended extensions (`.vscode/extensions.json`).

1. Run `uv sync` once. It creates `.venv`, which includes `ipykernel` and `jupytext`.
2. Open a notebook's `.py` file. VS Code picks `.venv/bin/python` (`.vscode/settings.json`).
3. Click **Run Cell** above a `# %%` line, or press Shift+Enter. The cells run in the Interactive Window, with outputs and plots.

Edit the `.py` file. It is the only copy in git.

VS Code does **not** keep a paired `.ipynb` in sync when you save. If you prefer the notebook view, run this first, open the `.ipynb`, and run the same command again before committing:

```bash
uv run jupytext --sync notebooks/*.py
```

The pre-commit hook also runs `jupytext --sync` on changed notebook `.py` files.

## In Jupyter

Run `uv run jupytext --sync notebooks/*.py` and open the `.ipynb`. Jupyter keeps the `.py` file in sync on save (configured in `jupytext.toml`), but only if `jupytext` is installed in the environment Jupyter runs from.

## What is needed

| Where | Package or extension | Why |
|---|---|---|
| `.venv` (dev group) | `ipykernel` | The kernel VS Code starts to run cells |
| `.venv` (dev group) | `jupytext` | Pairs the `.py` files with `.ipynb` |
| VS Code | Python (`ms-python.python`), Pylance, Python Debugger | Interpreter, editing and debugging |
| VS Code | Jupyter (`ms-toolsai.jupyter`) | Interactive Window and notebook view |
| VS Code (optional) | Ruff, Even Better TOML, Verilog-HDL | Linting and formatting, TOML files, `rtl/` |

## Tested setup

| Item | Version |
|---|---|
| VS Code | 1.144.0, Linux |
| Python extension / Jupyter extension | 2026.10.0 / 2025.9.1 |
| Python, ipykernel, jupytext | 3.12.3, 7.4.0, 1.19.6 |

`00_verification_metrics.py` was paired with `jupytext --sync` and all of its cells ran in an ipykernel started from `.venv` (2026-11-11).
