# Golden vectors

Small input/output pairs (`.npy` / `.npz`) produced by the Python reference of each kernel.
They are committed so verification is reproducible without downloading recordings.

- One folder per kernel: `tests/golden/<kernel>/`.
- Keep each file small (aim for under 1 MB). Record how it was generated in a `README.md` next to it.
- Each kernel's input excerpts, quantization and tolerances live in `tests/golden/<kernel>/config.toml`.
- Input excerpts taken from public recordings keep their own license. Give the source, the license and the changes in the folder's `README.md`, and add a row to `THIRD_PARTY.md`.
