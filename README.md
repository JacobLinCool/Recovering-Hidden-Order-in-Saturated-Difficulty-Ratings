# Recovering Hidden Order in Saturated Difficulty Ratings

This repository contains the ICASSP 2027 manuscript, the code for its global and sequence experiments, and the frozen public evidence used in the paper.

## Contents

- `paper/icassp2027/main.tex`, `references.bib`, the conference style files, the three figure sources or outputs, and `main.pdf` are the submission manuscript and build inputs.
- `paper/icassp2027/artifact/public/` contains the released chart-level out-of-fold scores, aggregate comparisons, model configurations, per-fold sequence run status, and title-free Dan-i Dojo matches. `MANIFEST.json` records file hashes and sizes.
- `TaikoChartEstimator/research/` and `scripts/research/` contain the study's feature extraction, models, training, analysis, and figure-generation code. The training and original release-building scripts need the restricted source charts and title-bearing experiment tables; those inputs are deliberately not redistributed.
- `tests/` and `scripts/verify_public_release.py` check the public package without those restricted inputs.

## Verify this release

From the repository root:

```bash
uv sync --frozen
uv run python scripts/verify_public_release.py
uv run python scripts/research/analyze_public_release.py
uv run pytest -q
cd paper/icassp2027
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The public verifier checks the exact manifest file set, hashes, privacy boundary, frozen row counts, and required manuscript inputs. The public analyzer independently recomputes the point estimates that can be derived from the released OOF and course rows. The LaTeX build checks that the manuscript can be compiled from files in this repository. The artifact's `REPRODUCE.md` states which analyses can be rerun from the released rows.

## Data boundary

The source chart dataset is access controlled. This repository does not distribute chart files, audio, raw chart identifiers or titles, title-bearing fold assignments, raw GPU checkpoints, or gameplay records. The published chart keys are deterministic pseudonyms, not a guarantee of anonymity; public course metadata can be matched with its public source tables. The underlying course facts retain their source attribution and are not relicensed by the code's MIT license.

The release starts at frozen out-of-fold predictions. Training from raw charts and exact title-cluster bootstrap intervals need restricted inputs. `reproducibility_inventory.json` preserves hashes and paths from the original experiment environment as historical provenance; it does not assert that those private inputs are present here. The retained experiment code has been scoped to the current paper, so its current hashes can differ from that historical inventory.
