# Lab 3 — Registry, FastAPI and Docker

## Answers

1. The initial registered food11 version and originating run are in experiments.json. A logged model belongs to one run; the registry gives a shared model name and independently numbered releases.
2. Aliases such as `champion` and `challenger` replace deprecated stages. These are user-defined names rather than mandatory built-in stages. Aliases can move between immutable model versions without changing serving code.
3. `models:/food11@champion` resolves the selected version through the registry. Move the alias and restart inference to serve another version; no .pth pathname or image rebuild is needed.
4. Dependency files are copied first so a source-only change reuses the expensive dependency installation layer. `--no-install-project` is required at this point because source/README have not been copied yet; uvicorn imports the source copied into the runtime stage.
5. The multi-stage image copies only the installed environment and source from the builder. A naive build also carries build tooling/cache. Exact measured image sizes/history, when available, are in verification.md; no guessed size comparison is presented as a measurement.
6. .dockerignore avoids transferring data, caches, Git history and MLflow state. Accidentally copying a Windows .venv over a Linux environment can break interpreter and library loading. Large datasets mainly increase context transfer and build time; they increase image size if copied into layers.
7. Container loopback belongs to that container. `host.docker.internal` is Docker Desktop's host gateway. Compose instead uses service DNS `mlflow`. Standalone localhost-only MLflow is used for local training; Compose hosts MLflow inside its own private service network.
8. A new container retrieves champion at startup. The model weights are in MLflow, whereas the image contains preprocessing, API code and dependencies. Check verification.md for actual runtime results and any local Docker limitation.
9. Publish an immutable image to a registry and pin its commit SHA tag (or digest). The Lab 5 workflow publishes ghcr.io/jerome397/mlops2:<git-sha> after successful tests and smoke checks.

The API has no import-time model download. Startup loads exactly once, tests can inject a fake model through SKIP_MODEL, unavailable models return 503, invalid images return 400, and output logits are converted to a stable softmax. CLASSES matches ImageFolder's alphabetical ordering.
