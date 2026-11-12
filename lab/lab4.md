# Lab 4 — Docker Compose

The root compose file builds MLflow, inference and Streamlit; persists tracking state in mlflow-data; uses health checks and service DNS; and publishes only MLflow/frontend on host loopback. **The older lab's references to Staging are interpreted as the current champion alias**, consistently with Labs 3 and 5.

## Answers

1. A stopped container retains its writable layer, but removing it discards that layer. A new standalone container without storage starts with an empty registry.
2. A named volume is Docker-managed and independent of container replacement. A bind mount also persists state, but ties deployment to host paths and host file permissions.
3. Compose connects services to a shared network with DNS entries named after services; inference can resolve `mlflow` there. Outside that network the name has no meaning.
4. INFERENCE_URL permits both local standalone development and Compose networking without editing/rebuilding source.
5. Frontend reaches inference:8000 over the private network; a host-published inference port is unnecessary. MLflow:5000 and frontend:8501 are the two published host ports.
6. Short-form depends_on only orders process startup. Model loading can fail before MLflow is ready. This implementation waits for MLflow health, and inference restarts on failure. An empty registry still needs a champion; health alone does not create one.
7. Expected compose ps: MLflow has host 5000, frontend host 8501, inference no host mapping. Actual local execution status is recorded honestly in verification.md.
8. Refreshing the frontend does not reload the model; inference keeps its startup instance. After moving champion, `docker compose restart inference` reloads it.
9. Restart works because weights are fetched from the registry at startup rather than baked into the image. Rebuild only for code/dependency changes.
10. `down`/`up` preserves the named volume and registry/alias. `down -v` removes that volume, losing metadata and artifacts. Do not perform that destructive experiment on the working registry; an empty new volume demonstrates the same storage boundary.
11. Three replicas need routing/load balancing and replica orchestration. Machine-failure survival also needs a highly available metadata database, shared durable artifact storage, backups and multi-node scheduling; single-machine Compose does not supply these guarantees.

## Clean startup

Stop standalone MLflow if it occupies 5000. Run `docker compose up -d --build mlflow`, train/register a champion against localhost:5000, then `docker compose up -d --build`. A fresh Compose volume and the standalone mlflow.db are separate registries. HTTP artifact serving avoids inaccessible host-only artifact paths inside inference.

## Executed integration results

[Successful Linux-runner check](https://github.com/Jerome397/MLOPS2/actions/runs/43229215608) built all three images and verified their health, actual HTTP image prediction, startup retrieval of a registered model, moving champion from version 1 to 2, restarting inference, and preserving version 2/predictions after a complete Compose down/up cycle. CI seeds an explicitly synthetic untrained model to test wiring; it does not train or claim Food-11 accuracy. The actual locally trained champion was separately loaded through FastAPI and tested with a validation upload (inference_check.json).

Docker Desktop initially failed with Windows error 1920 on stale runtime sockets. A metadata-only inspection proved the affected directories contained only zero-byte IPC sockets, without credentials. A guarded, reversible backup of those exact runtime directories was then approved and restored engine 29.4.0. No factory reset, volume purge or deletion of persistent Docker data was performed.

## Local side-by-side deployment

The localhost training registry remains on 5000 and Python frontend on 8502. To run Compose alongside them:

```powershell
docker compose -p food11-labs -f docker-compose.yml -f docker-compose.local.yml up -d --build --wait mlflow
uv run python src/food11/migrate_registry.py
docker compose -p food11-labs -f docker-compose.yml -f docker-compose.local.yml up -d --build --wait
```

The override publishes the Compose registry on 5001 and frontend on 8503. Migration copies the actual trained champion and its metrics into the separate Compose registry, recording its original run/version as provenance tags. It does not retrain or replace the original training registry.

The real local stack passed the same lifecycle checks after migration: a frontend-container request returned Bread at 96.8192% confidence; moving champion to version 2 and restarting inference preserved prediction; down/up retained the alias, artifacts and prediction. See compose_local.json. The local MLflow allowlist includes both 5000 and 5001 so the supplied port override works.
