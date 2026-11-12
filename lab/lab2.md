# Lab 2 — Training and MLflow

The script uses pretrained ResNet18, replaces its head with 11 outputs, loads ImageFolder splits, supports all required hyperparameters, and logs epoch training/validation metrics, final test accuracy and the actual trained model. Optional frozen-backbone transfer learning makes the controlled CPU comparison practical. All compared runs use the same seed, dataset and training mode.

## Answers

1. pyproject.toml declares MLflow/PyTorch/torchvision/scikit-learn and other runtime dependencies. uv.lock pins their versions and artifacts for reproducible resolution. PyTorch uses the explicit CPU index.
2. `--backend-store-uri` selects the metadata database (SQLite here): experiments, runs, parameters, metrics and registry records. Artifacts are larger files such as model weights. `--artifacts-destination ./mlruns` stores them on disk while `--serve-artifacts` proxies access over HTTP. This is necessary when a client container cannot read host filesystem paths.
3. mlflow.db and mlruns are mutable tracking state, not code or a dataset snapshot. Git/DVC are poor concurrent databases. They are ignored; production deployments should back up the database and artifact storage separately.
4. `set_experiment("food11")` creates the experiment if absent, then directs new runs into it.
5. Parameters describe a run's fixed configuration. Metrics are numerical observations recorded repeatedly; `step=epoch` aligns their learning curves.
6. Params, metric history and the model are visible in the run. Artifacts live under this project's mlruns directory and are retrieved through MLflow's artifact endpoint. MLflow 3 may organize logged models under model-specific IDs rather than an older run/artifacts/model filesystem layout.
7. See experiments.json for the controlled learning-rate results. Higher learning rate is not necessarily better: large updates can overshoot useful weights.
8. The comparison varies one of learning rate/batch size at a time; compare val_accuracy with those axes. The measured best settings are summarized in verification.md. This small dataset does not support a universal conclusion about learning rates.
9. The best **final validation accuracy** among the controlled runs selects the initial champion. Its run ID and registered version are recorded in experiments.json. Test accuracy is reported independently and is not used to choose this initial best run.

The first full-fine-tuning run achieved final validation accuracy 0.5000 and test accuracy 0.510036. A Windows terminal emoji encoding error occurred while closing that run, after the model and metrics had been saved. The console encoding was fixed and that completed run was marked finished; it is separate from the controlled frozen-backbone comparison.
