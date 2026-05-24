# Archive Report: Fix HRR Dimension Sync

## Status

Archived.

This change is a retroactive OpenSpec record for a small production bugfix already committed and pushed to `main`.

## Source Issue

- GitHub issue: [#20 — HRR dimension mismatch between Store/Retriever defaults and stored vectors](https://github.com/Basiliskode/memory-etch/issues/20)
- Author: `Hermie-cell`
- Labels: `bug`, `priority:high`

## Root Cause

`EtchRetriever` defaulted to `hrr_dim=256` independently from the store. Existing or custom stores may contain HRR vectors with a different dimension, commonly `1024`. The retriever encoded query vectors at 256 and then attempted similarity against stored vectors at another dimension.

The previous `_score_candidates()` path caught all exceptions with `except Exception: pass`, so HRR failures were hidden and retrieval silently degraded to FTS5/Jaccard.

## Implemented Repair

Commit: `8c3e2fa fix: sync retriever HRR dimension with store`

### Code Changes

- `src/memory_etch/store.py`
  - Added public `get_effective_hrr_dim()`.
  - Added public `compute_hrr_batch()` compatibility wrapper.
- `src/memory_etch/retrieval.py`
  - Changed `EtchRetriever.__init__(hrr_dim: Optional[int] = None)`.
  - Uses `store.get_effective_hrr_dim()` when no explicit dimension is provided.
  - Logs dimension mismatch warnings and unexpected HRR exceptions instead of silently swallowing them.
- `tests/test_retrieval.py`
  - Added `TestRetrieverHRRDimensions` regression tests.

## Verification

Commands already run:

```text
py -3.11 -m pytest tests/test_retrieval.py::TestRetrieverHRRDimensions -v --tb=short
py -3.11 -m pytest tests/test_retrieval.py tests/test_hrr.py -v --tb=short
py -3.11 -m pytest tests/test_store.py tests/test_etch_v2.py -v --tb=short
```

Results:

- `TestRetrieverHRRDimensions`: passed.
- Retrieval + HRR suite: 36 passed.
- Store + Etch v2 suite: 87 passed.

Known local full-suite issues were unrelated: missing optional `mcp`/`fastembed` dependencies and pre-existing CI workflow expectation mismatch.

## Issue Closure

Issue #20 was closed after archive confirmation.

Closure comment summarized:

- Retriever now syncs HRR dimension from the store.
- Store exposes public HRR dimension and batch flush APIs.
- HRR no longer fails silently on dimension mismatch.
- Regression coverage exists for 1024-dim store plus default retriever.

## Risks

No active risks.

The behavior change only affects omitted retriever dimensions. Explicit `hrr_dim` callers remain supported.

## Final Outcome

The repair is complete, pushed to `main`, documented in OpenSpec, and GitHub issue #20 is closed.
