# Public reproduction contract

From the repository root, install the locked environment and verify the released files:

```bash
uv sync --frozen
uv run python scripts/verify_public_release.py
uv run python scripts/research/analyze_public_release.py
```

The verifier uses only this repository. It checks the exact file set and SHA-256 hashes in `MANIFEST.json`, required paper assets, chart and run counts, and that public rows have no direct title, chart-ID, player, audio, or raw-chart fields. The checked-in `main.tex` can be built with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `paper/manuscript/`.

The released out-of-fold scores let readers inspect per-chart predictions and independently recompute paper point estimates; the public analyzer checks those values against the frozen evidence and manuscript macros. The Dan-i Dojo figure can be regenerated with `uv run python -m scripts.research.make_dojo_criterion_figure` from the released title-free rows. Cosmetic jitter is derived from published pseudonyms, so the rebuilt dots need not have byte-identical horizontal positions. Aggregate tables also contain the reported confidence intervals, per-seed metrics, and the lower-rated Dojo control. Exact reruns of title-cluster bootstraps, the lower-rated control's row-level calculation, sequence per-seed chart predictions, and training require restricted inputs that are not redistributed here. The original `analyze_*`, `build_*`, and training scripts are included for method inspection and use with those authorized inputs; running them with their original default data paths is not part of this public contract. The source hashes in `reproducibility_inventory.json` identify the original experiment snapshot, not the curated code in this repository.
