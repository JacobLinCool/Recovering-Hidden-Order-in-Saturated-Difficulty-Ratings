# Public experiment specification

## Research question

Can a model recover order that an ordinal scale erases at its maximum, and
does the recovered order agree with an independent expert curriculum?

## Chart task

- Corpus: 1,023 Oni charts in 1,019 Unicode-normalized title groups.
- Label: integer official star rating from 1 to 10.
- Artificial scenarios: `C_c(y)=min(y,c)` for caps 7, 8, and 9.
- Leakage unit: normalized title.
- Five outer folds; each chart has exactly one OOF prediction.
- Fitting, inner validation, early stopping, and model selection see only the
  observed capped labels. Intact labels are outer-test evaluation only.
- Primary hidden-tail metrics: Spearman rho, Kendall tau-b, and strict accuracy
  over pairs with unequal intact labels. Prediction ties receive no credit.
- Uncertainty: 10,000 paired normalized-title bootstrap replicates.

## Global conditions

Nineteen duration, density, inter-onset, tempo, scroll, and note-type features
feed ordinary Huber, one-sided Huber, Ridge, GBDT, cumulative logit/probit, and
extended-binary ordinal reduction. Cap 8 is primary; caps 7 and 9 are frozen
sensitivities. Inner observed-label MAE selects each condition's standard
hyperparameters.

## Sequence control

Each event contains drum-action type, beat position, duration, BPM, scroll, and
local density. A four-layer 256-dimensional Transformer uses masked token and
window means. Cap 8 contains ordinary Huber, one-sided Huber, and ordinal logit
over five folds and seeds 2025--2027. Caps 7/9 contain frozen single-seed
ordinary/ordinal sensitivities. Native ordinal OOF uses three seeds. The public run-status table covers 80 successful CUDA fold runs; there is
no fallback estimator or substituted seed.

## Independent curriculum criterion

- Source: public community transcriptions of official 2020--2025 Dan-i Dojo
  course tables, corroborated by official mode documentation and annual
  notices. Retrieval bytes and source metadata are SHA-256 identified.
- Primary panel: Oni, official star 10, expert tiers 10th Dan, Kuroto, Meijin,
  Chojin, and Tatsujin.
- Join: exact normalized title only; exact display title resolves collisions;
  otherwise the placement is excluded. Fuzzy matching is forbidden.
- Frozen result panel: 47 unique placements across six editions; three of 50
  eligible placements are unmatched under the strict rule.
- Estimands: macro mean of yearly Spearman correlations and macro yearly
  strict-pair concordance for unequal tiers.
- Uncertainty: 10,000 hierarchical bootstrap replicates over years and then
  placements, plus 10,000 within-year blocked permutations.
- Replication: native sequence ordinal index on the identical 47 placements.
- Control: equally rated within-year pairs below star 10, reported separately
  and descriptively.
- Course position is exploratory and is not assumed to encode difficulty.

## Interpretation boundary

The recovered output is a relative ordinal content index, not a calibrated
extension of the official star scale. Intact stars and curriculum tiers are
design decisions, not continuous psychophysical ground truth. This study uses
no human participants, player profiles, or gameplay records. The public
artifact redistributes no chart file, title, audio, or proprietary asset.
