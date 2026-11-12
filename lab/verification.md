# Verification — 12 November 2026

## Completed checks

- Restored 16,643 original Food-11 images, preserving the old project under Documents.
- Rebuilt the full processed dataset (16,032) and mini dataset (3,292), excluding 611 cross-split pixel duplicates while preserving raw data.
- Both real processed datasets passed all five data-contract checks.
- A deliberate duplicate in real mini data caused exit 1; the copy was removed automatically (duplicate_check.txt).
- DVC pointer tracks 35,967 paths, 1,377,563,963 bytes. `dvc push` backed up 32,034 unique objects and `dvc status --cloud` reported cache/remote in sync.
- The Git/DVC raw-only versus processed-snapshot checkout demonstration passed in the temporary clone (dvc_checkout.json).
- A new Git clone initially had no data; `dvc pull` fetched 32,034 objects and restored all 35,967 paths. See dvc_restore.json.
- Lint passes and 15 unit/API/contract/promotion-boundary tests pass without a live registry, GPU or DVC dataset.
- Seven real five-epoch training runs completed: one full-fine-tuning run, four controlled frozen-backbone comparison runs and two promotion candidates. Actual model weights/metadata live in MLflow, not Git.
- Comparison winner: learning rate 0.001, batch 32, validation accuracy 67.7920%, evaluation accuracy 73.1752% on 1,096 images. Run 48a88dead297420dbe52bdf7f58d4ee8 is food11 version 1, alias champion.
- Candidate version 2 scored 12.8650%; version 3 scored 73.1752%. Both became challenger because promotion required 75.8517%; champion remained version 1. Full numbers and tags: experiments.json.
- FastAPI startup retrieved models:/food11@champion through MLflow's artifact server; /health returned ok and an uploaded image returned HTTP 200 with a category/confidence (inference_check.json). This proves the serving path, not perfect classification: the recorded Bread image was misclassified as Fried food.
- GitHub main requires PRs and test/smoke checks. PR #1 deliberately failed, showed a blocked merge, then was fixed (branch_protection.json).
- GitHub built the API, MLflow and frontend images, verified real HTTP prediction with a synthetic smoke model, moved its champion alias, restarted inference and verified persistence through Compose down/up. [Successful run](https://github.com/Jerome397/MLOPS2/actions/runs/43229215608): test 40 seconds; smoke 325 seconds. See ci_results.json. Publication is restricted to successful main pushes.

## Local services

Run `powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1`. It starts one localhost MLflow worker, waits for readiness, starts inference and then the frontend. Existing listeners are left unchanged.

- MLflow: http://127.0.0.1:5000
- API documentation: http://127.0.0.1:8000/docs
- Frontend: http://127.0.0.1:8502 (8501 already belongs to another project)

## Machine limitation

Docker Desktop fails before its Linux engine starts with a Windows error 1920 on stale runtime socket objects, then the secrets-engine socket. Automatic approval review rejected moving the secrets-engine directory because of its shared secrets/credential role. It was left untouched; persistent Docker data was not reset or purged. Container and Compose behavior was verified on GitHub's Linux runner instead. A manual Docker repair/restart may still be needed for local Compose operation.

## Storage and reproducibility

The permitted local DVC remote is C:/Users/user/OneDrive/Desktop/MLOPS1-dvc-storage, outside this Git repository. It is local storage, not publicly reachable DagsHub storage. A different machine must copy that remote or configure another accessible remote before dvc pull. MLflow state is deliberately ignored by Git/DVC; retain mlflow.db and mlruns together to retain the locally trained registry. CPU-only requirements are locked in uv.lock. Reports distinguish synthetic CI integration checks from real local model evaluation.
