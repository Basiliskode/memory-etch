# Proposal: Fix HRR Dimension Sync

## Intent

Formalize the repair for GitHub issue #20: HRR retrieval must be dimension-safe when a store contains vectors created with a dimension different from the retriever default.

The bug was already fixed in `main` by commit `8c3e2fa`, but this OpenSpec record preserves the root cause, scope, verification, and closure trail.

## Scope

### In Scope

- Make `EtchRetriever` use the store's effective HRR dimension when `hrr_dim` is omitted.
- Expose `EtchStore.get_effective_hrr_dim()` as the public source of truth for retrieval/integration code.
- Expose `EtchStore.compute_hrr_batch()` as a public compatibility wrapper for integrations that need to force pending HRR computation.
- Replace silent HRR similarity failures with warnings/logging.
- Add regression coverage for a 1024-dim store searched through a default retriever.
- Close GitHub issue #20 after verification.

### Out of Scope

- Changing the global default HRR dimension.
- Migrating existing vector blobs.
- Changing embedding/vector database behavior.
- Changing Hermes Agent code outside memory-etch.

## Capabilities

### New Capabilities

- `hrr-dimension-safety`: Retrieval uses the store's effective HRR vector dimension and does not silently hide dimension mismatch failures.

### Modified Capabilities

- None.

## Approach

Use `EtchStore` as the source of truth for HRR dimensions. When `EtchRetriever` is created without an explicit `hrr_dim`, it calls `store.get_effective_hrr_dim()` and encodes query vectors at that dimension. `_score_candidates()` checks vector lengths before similarity and logs mismatches rather than swallowing exceptions.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/memory_etch/retrieval.py` | Modified | Sync retriever HRR dimension from store by default; log mismatch failures. |
| `src/memory_etch/store.py` | Modified | Add public HRR dimension and batch computation methods. |
| `tests/test_retrieval.py` | Modified | Add regression tests for issue #20. |
| GitHub issue #20 | Closed | Document repair and verification. |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Explicit callers relied on retriever default `256` | Low | Explicit `hrr_dim=256` still works. Only omitted value now syncs from store. |
| Logging exposes previously silent mismatch | Low | This is intentional and improves diagnostics. |
| Legacy DB has mixed dimensions | Low | Mismatched candidates are skipped with warning instead of crashing or silently failing. |

## Rollback Plan

Revert commit `8c3e2fa`. This restores the previous retriever behavior, but it also reintroduces silent HRR degradation for existing/custom DBs with non-default vector dimensions.

## Dependencies

- Existing HRR implementation in `src/memory_etch/hrr.py`.
- Existing store vector persistence and `_get_effective_hrr_dim()` behavior.

## Success Criteria

- [x] `EtchRetriever(store)` uses `store.get_effective_hrr_dim()` when `hrr_dim` is omitted.
- [x] A 1024-dim store searched with default retriever produces `_hrr_sim > 0`.
- [x] `EtchStore.compute_hrr_batch()` exists and flushes pending HRR vectors.
- [x] HRR dimension mismatches are logged instead of silently swallowed.
- [x] Focused regression tests pass.
- [x] GitHub issue #20 is closed.
