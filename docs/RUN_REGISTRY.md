# Run Registry

The local SQLite registry stores run identity, status, manifest/config hashes, parent lineage, metrics, decisions, and artifact checksums.

```bash
wai-r0 runs init --database reports/runs.sqlite
wai-r0 runs register --database reports/runs.sqlite reports/run/report.json
wai-r0 runs list --database reports/runs.sqlite
wai-r0 runs show --database reports/runs.sqlite RUN_ID
```

SQLite uses transactional writes and durability settings appropriate for local research metadata. The registry is an index, not the sole copy of artifacts. Deleting a checkpoint or dataset can make a registered run non-reproducible; artifact existence and hashes must still be verified.
