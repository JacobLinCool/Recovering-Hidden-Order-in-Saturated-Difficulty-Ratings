# Public Artifact Privacy Contract

## Release boundary

This release uses no human participants, player records, interactions, or
interventions. Its active global-analysis inputs are exactly two frozen,
title-grouped out-of-fold tables: fold assignments and model scores. The
sequence control likewise begins at four hash-pinned OOF tables: fold runs,
seed scores, ensemble scores, and aggregate metrics. The Dojo criterion is
public course metadata designed independently of the models.

The public artifact may contain source code, frozen configurations, derived
chart-level out-of-fold scores, aggregate model metrics, title-free Dojo
criterion rows, repository-relative source inventories, SHA-256 digests, and
commands that regenerate paper exhibits from canonical evidence.

It must not contain raw chart files, audio, account data, gameplay records,
secrets, absolute workspace paths, or any additional input outside the release
manifests. The public contract does not hash or inventory excluded data.

## Automated gate

The builder and submission verifier both fail closed on:

- structured field names that identify player-level data;
- path-like values or documentation lines that reference player-data sources;
- secrets, tokens, private-key material, or absolute user paths;
- missing, extra, stale, or hash-mismatched files in the public manifest.

Natural-language statements describing this privacy boundary are allowed.
Mutation tests exercise both forbidden structured keys and forbidden paths.
There is no compatibility switch or permissive fallback.
