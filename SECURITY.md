# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security vulnerabilities.

Use GitHub's private vulnerability reporting flow for this repository when it is available. If private reporting is not enabled yet, open a minimal public issue that asks for a private contact path; do not include exploit details, credentials, database contents, or proof-of-concept payloads in that public issue.

## What to include

- A short summary of the issue.
- Affected version or commit.
- Reproduction steps or proof of concept.
- Expected impact, including whether local files, SQLite data, or credentials are exposed.
- Any suggested fix, if you have one.

## Response expectations

This is a small open-source project, so response times are best-effort. Maintainers will acknowledge credible reports, investigate, and coordinate disclosure before public details are shared.

The maintainer should enable GitHub private vulnerability reporting before the first public release so reporters have a clear private channel.

## Scope

Memory Etch is local-first. Reports involving local file handling, SQLite storage, plugin/provider boundaries, and accidental secret exposure are in scope. SaaS, hosted auth, and multi-tenant issues are out of scope unless the repository adds those features later.
