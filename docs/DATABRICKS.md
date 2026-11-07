# Databricks notes

Moved from AGENTS.md §10 (this project doesn't use Databricks yet). Read this if it starts to.

- Use **Databricks Git folders (Repos)** connected to this repo; work on your feature branch inside it, never edit `main` in the workspace.
- Store notebooks in **source format** (`.py` with `# COMMAND ----------` / `# MAGIC %sql`) so diffs and reviews work. Clear outputs before committing `.ipynb`.
- **Always use 3-level names**: `catalog.schema.table`. Agree on one catalog/schema per team (e.g. `hack.team_x`), and per-person dev schemas for experiments (`hack.dev_<name>`).
- **Medallion layout**: `bronze_*` (raw, append-only), `silver_*` (cleaned, typed, deduped), `gold_*` (aggregates for dashboards/ML). Name tables accordingly.
- Never `DROP`/`TRUNCATE`/`OVERWRITE` a shared `silver_`/`gold_` table without telling the team. Delta time travel (`RESTORE TABLE t TO VERSION AS OF n`) is the undo button.
- Avoid `.toPandas()` / `.collect()` on large DataFrames; sample or aggregate first.
- Log ML runs with **MLflow** (`mlflow.autolog()`); put the best run's link in the PR.
- Data quality checks (row counts, nulls, uniqueness) belong in the pipeline/notebook and count as the "test" for data changes (§2 Step 4).
- Never hardcode tokens: use `dbutils.secrets.get(scope, key)` or env vars.
- Shut down/auto-terminate clusters; prefer serverless where available.
