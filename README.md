# Food-11 MLOps — Labs 1–5

Repaired project for Jerome397/MLOPS2. Data and trained models stay outside Git.

## Setup

```powershell
uv sync --frozen
uv run dvc pull
uv run python src/food11/validate_data.py data/food11_processed_mini
uv run ruff check src tests frontend
uv run pytest -q
```

Python 3.13, CPU-only PyTorch, dependencies pinned in uv.lock. uv uses file copies
to avoid OneDrive hardlink errors.

## Data

`data/food11_raw/{training,validation,evaluation}` preserves original images.
`data/food11_processed` has RGB 128×128 ImageFolder data. The mini dataset has
up to 100 images per class per split. Cross-split duplicates are excluded from
processed data, preserving evaluation before validation before training.
See `lab/data_preparation.json` for counts and excluded paths.

The lab-approved local DVC remote is outside Git at
`C:/Users/user/OneDrive/Desktop/MLOPS1-dvc-storage`. On another machine configure
an accessible remote (`uv run dvc remote modify local <path>`). This local backup
is not a public download service. No credentials are committed.
To rebuild, archive existing processed directories, then run:

```powershell
uv run python src/food11/data.py
uv run dvc add data
uv run dvc push
```

## Training and registry

Start MLflow in a separate terminal:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./mlruns
```

Then train, register, serve:

```powershell
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32 --freeze-backbone
uv run python src/food11/register.py --run-id <run-id> --champion
uv run uvicorn src.food11.serve:app --host 127.0.0.1 --port 8000
```

`--freeze-backbone` trains the 11-class head on pretrained ResNet18 features;
omit it to fine-tune the entire network. Training validates data and logs seed,
dataset fingerprint, evaluation size, hyperparameters, epoch metrics and model.
The API uses matching preprocessing and class ordering. It loads
`models:/food11@champion` once at startup. Restart after changing the alias.
`SKIP_MODEL=1` permits isolated health checks; predictions then return 503.

## Docker Compose

Stop standalone MLflow before starting the Compose registry on the same port.

```powershell
docker compose up -d --build mlflow
# Train and register a champion against localhost:5000, then:
docker compose up -d --build
```

The frontend is [localhost:8501](http://127.0.0.1:8501), MLflow is
[localhost:5000](http://127.0.0.1:5000). Inference has no host-published port.
The Compose registry starts empty and is separate from standalone MLflow.
Register a champion before starting inference. Service DNS and health checks
handle readiness. MLflow serves artifacts over HTTP so client containers need
no access to the server's filesystem. The named `mlflow-data` volume persists
metadata and artifacts. `docker compose down` retains it; `down -v` destroys it.

## CI/CD

GitHub Actions runs lint/tests, builds the API and probes health with SKIP_MODEL.
Only successful pushes on main publish `ghcr.io/jerome397/mlops2:<commit-sha>`.
Code images and the registry's champion model version are released independently.

```powershell
uv run python src/food11/promote.py --version <candidate-version>
```

The promotion gate checks logged dataset fingerprints/test sizes and requires
candidate accuracy >= champion accuracy + two binomial standard errors.
Decisions are written as model-version tags. Run it where MLflow and evaluation
data are accessible. See `lab/lab1.md`–`lab/lab5.md` and `lab/verification.md`
for answers, measured results and machine-specific limitations.
