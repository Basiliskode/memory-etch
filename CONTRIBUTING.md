# Contributing to memento

Thanks for helping improve Memento. Keep changes small, local-first, and easy to review.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before you code

- Open or link an approved issue for user-visible changes.
- Keep the scope focused: one problem, one PR.
- Avoid new mandatory runtime dependencies unless the issue explicitly approves them.
- Do not claim production stability beyond what the current tests and docs support.

## Test workflow

Run the focused tests for your change first:

```bash
python -m pytest tests/test_store.py
```

Then run the broader suite before opening a PR:

```bash
python -m pytest tests/
```

Hermes-dependent E2E tests are not part of the default workflow. Run them manually only when your change touches provider/plugin integration.

## Ruff workflow

Check linting without mutating files:

```bash
python -m ruff check src/ tests/ --no-fix
```

Check formatting:

```bash
python -m ruff format src/ tests/ --check
```

If formatting is required, run `python -m ruff format src/ tests/` and review the diff.

## Commit convention

Use concise conventional commits:

- `feat: add retriever option`
- `fix: handle empty FTS query`
- `docs: clarify local setup`
- `test: cover changelog structure`
- `chore: update ci workflow`

## Changelog guidance

Update `CHANGELOG.md` for user-visible changes. If no changelog entry is needed, say why in the PR checklist.

## Pull request checklist

- [ ] Linked issue included (`Closes #...`) or explained why none is needed.
- [ ] Focused tests added or updated.
- [ ] `python -m pytest tests/` passes locally, or failures are documented.
- [ ] `python -m ruff check src/ tests/ --no-fix` was run, or known baseline debt is documented.
- [ ] `python -m ruff format src/ tests/ --check` was run.
- [ ] `CHANGELOG.md` updated or marked not needed.
- [ ] Docs avoid overstating stability or production readiness.
