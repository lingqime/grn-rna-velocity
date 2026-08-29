# Public release audit

Automated packaging checks performed on this release candidate:

- canonical machine-specific absolute-path hits: 1
- obvious credential/private-key pattern hits in entire repository: 0
- canonical Python syntax errors: 0
- repository files larger than 10 MiB: 0

The `archive/` directory intentionally preserves historical notebook code and
may contain old machine-specific paths. Those paths are provenance, not
canonical execution paths.

Remaining owner decisions are listed in `PUBLIC_RELEASE_CHECKLIST.md`.
