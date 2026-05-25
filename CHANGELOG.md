# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Community contribution docs, issue templates, PR template, and repository text normalization.

### Fixed

- Aligned runtime package version, README snippets, MCP docs, extras docs, and release validation checks with the 1.1.0 package metadata.

## [1.1.0]

### Added

- Hive Memory provenance fields with governed scopes and inbox review lifecycle.
- MCP inbox review tools: `list_inbox`, `promote_fact`, and `reject_fact`.

## [1.0.0]

### Added

- MCP stdio server integration for agent workflows.
- Structured fact fields (`what`, `why`, `where`, `learned`) and automatic project detection.
- Pluggable embedding providers, expanded search fallback, HRR multi-query search, dynamic RRF, exact deduplication, conflict surfacing, circuit breaker protection, auto-eviction, session summaries, and progressive disclosure.

## [0.2.0]

### Added

- SQLite-backed memory store with FTS5 search and optional HRR vector support.
- Retrieval and query classification helpers for local-first agent memory workflows.
- Optional BGE-M3/fastembed integration behind extras.

### Changed

- Package metadata, CI, and quality tooling prepared for broader open-source contribution.

## [0.1.0]

### Added

- Initial Memory Etch package foundation.
- Basic persistent memory APIs and local development workflow.
