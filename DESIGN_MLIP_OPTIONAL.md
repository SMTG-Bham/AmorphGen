# Design: MLIP-Optional AmorphGen

**Status**: agreed 2026-07-31 · partially implemented (see checklist)
**Goal**: the base `pip install amorphgen` has **zero ML dependencies** —
random generation, full numeric + plotted analysis, and classical-potential
pipelines work out of the box (~80 MB). MLIP backends (MACE, CHGNet,
SevenNet) are opt-in extras that pull in PyTorch transitively.

## Motivation

- Most entry-point tasks (generate seeds, analyse trajectories) need no MLIP.
- A ~2 GB torch download for a "structure generation" install is a common
  JOSS-review complaint and blocks analysis-only users on clusters/laptops.
- The calculator factory (`get_calculator`) is already a clean boundary; the
  restructure cashes in existing architecture rather than redesigning it.

## Decisions

### D1 — Packaging: single package + extras
One PyPI package. Base deps: `ase, numpy, scipy, pyyaml, matplotlib`
(deliberately **no torch**). Extras: `[mace]`, `[chgnet]`, `[sevennet]`,
bundles `[all]`/`[full]`, plus `[dev]`, `[docs]`.
*Rejected*: two-package split (amorphgen-core + meta) — doubles release
burden, muddies the JOSS submission; lazy `__getattr__` subpackage gating —
engineering for marginal gain.

### D2 — Failure UX: fail fast at CLI dispatch
Modes that require a calculator (pipeline, `--relax`, `--batch-opt`,
`--batch-quench`, ensembles) abort **before any setup work** when the
requested model's backend is missing, with a curated message: installed vs
missing backends, the exact `pip install "amorphgen[...]"` line, and the
classical-potential alternative.
**Single source of truth**: the CLI check calls a shared helper in
`utils/calculators.py` (e.g. `available_backends()` / `require_backend(model)`)
— backend knowledge is not duplicated in the CLI. The factory's existing
lazy ImportError messages remain as the API-level backstop.
*Rejected*: silent auto-fallback to classical — an MLIP→LJ swap changes the
science; it must stay an explicit user choice (the fail-fast message may
*suggest* it).

### D3 — Default model: fixed `mace-mpa-0`
`DEFAULT_CONFIG` keeps a fixed default. A config without an explicit model
means the same physics on every machine; bare/chgnet-only installs get the
fail-fast with guidance instead of a silent backend switch.
*Rejected*: auto-pick first installed MLIP — the same command would run
different physics depending on the environment (reproducibility trap).

### D4 — `--list-models`: full registry + installed markers
On every install, show all known models with per-backend markers, e.g.
`[installed]` / `[not installed -> pip install "amorphgen[mace]"]`. Serves discovery
("what could I use?") and diagnosis ("why did it fail?") at once.

### D5 — `utils.radii` stays in the base
Pure data + numpy; powers random-gen. The existing `radii_data.json` export
is the documented integration point for external tools.
*Rejected*: standalone mini-package (second release cycle, heavy for a
one-maintainer project pre-JOSS).

### D6 — matplotlib moves into base; `[analysis]` extra retired
Numeric analysis is already torch-free in base; moving matplotlib (~30 MB,
no ML baggage) makes `--analyse --save-plot` work on a bare install and
removes the now-misleading extra name. Keep `[analysis]` as a deprecated
no-op alias for one release, then drop.

### D7 — CI: light matrix + weekly CHGNet canary
- **Every push/PR**: existing 3.10/3.11/3.12 matrix (Tier 1+2, torch-free)
  + the `light-install` job (bare install, asserts torch absent, smoke-tests
  generate/analyse/`device auto→cpu`/CLI). This job **is the contract** —
  any stray module-level `import torch` fails CI.
- **Weekly cron**: a canary job installs `[chgnet]` (CPU-capable, smallest
  download) and runs one tiny relax smoke — catches upstream API/pin
  breakage (e.g. the e3nn 0.4/0.5 split) without slowing PRs. MACE stays
  local-only pre-release (foundation-model downloads are large/flaky in CI).
*Rejected*: per-push backend jobs (download flakiness becomes our CI
flakiness); tags-only validation (main can drift broken for weeks).

### D8 — Docs lead with a task-based install matrix
README/docs/JOSS paper open with a 3-row table:

| I want to… | Install |
|---|---|
| generate structures / analyse trajectories | `pip install amorphgen` |
| MLIP relaxation & melt-quench MD | `pip install "amorphgen[mace]"` (or `[chgnet]`) |
| everything (MACE + CHGNet) | `pip install "amorphgen[all]"` |

Each user copies the right line first time; the zero-ML design is showcased,
not buried. SevenNet keeps its separate-env note (e3nn pin conflict).

## Invariants (enforced)

1. `import amorphgen` and all of: `generate_random`, `StructureAnalyser`
   (numeric + plots), classical LJ/Buckingham, CLI parsing — succeed with no
   torch installed. (CI: `light-install` job.)
2. `device="auto"` resolves to `"cpu"` when torch is absent
   (`utils.common.resolve_device`) — never raises.
3. Every torch import in the package is lazy (function-scope) and either
   guarded or only reachable behind an installed backend.
4. Error messages for missing backends always contain a copy-pasteable
   install command.

## Implementation checklist

Done (2026-07-31):
- [x] `torch` removed from core deps (`pyproject.toml`)
- [x] `resolve_device()` helper; 6 pipeline device-auto sites patched
- [x] Guarded torch import in `classical.py` GPU path (helpful message)
- [x] `light-install` CI job (bare install + torch-absence + smoke)
- [x] README light-install section
- [x] 3 unit tests for `resolve_device` (incl. no-torch fallback)

Done (same day):
- [x] D2: `available_backends()` / `require_backend()` in
      `utils/calculators.py`; CLI `_requires_calculator()` gate fails fast
      before any setup work (verified: no work dir created, exit 1)
- [x] D4: `--list-models` per-backend `[installed]` / install-hint markers
- [x] D6: matplotlib in core deps; `[analysis]` kept as deprecated no-op alias
- [x] D7: `.github/workflows/canary.yml` — weekly CHGNet relax smoke
- [x] D8: task-based install matrix in README + docs installation page
- [x] Tests: 9 new (fail-fast message contents, markers, mode gating)

Remaining:
- [ ] Paper (`paper/paper.md`) install line — align with D8 wording at
      next paper edit
