# Lab 1 — Git, DVC and data preparation

The existing GitHub repository is Jerome397/MLOPS2 (retained instead of creating a differently named repository). The original project and data under Documents were preserved. The empty desktop workspace now contains a clone of that repository and the restored raw data.

## Answers

1. `uv init` creates project metadata/dependencies in pyproject.toml, a Python version selection, a README and starter source. uv.lock records exact dependency resolutions; .venv is the installed environment.
2. `.dvc/config` stores non-secret repository settings; `.dvc/.gitignore` excludes local cache/config. `.dvcignore` excludes paths from DVC traversal. Commit configuration and ignore rules, not cache objects.
3. DVC global credentials live in the user's DVC configuration; `--local` writes ignored `.dvc/config.local`, while default project configuration is committed. `--system` is system-wide. Never commit passwords or tokens. No new credentials were needed here.
4. `dvc add data` adds `/data` to Git's ignore rules. Git tracks the pointer rather than the large images.
5. `data.dvc` records the directory MD5, total size, file count and workspace path. The DVC directory manifest identifies individual cached files.
6. GitHub contains code/configuration and data.dvc, not the dataset. **Selected permitted alternative: local DVC remote outside Git**, at `C:/Users/user/OneDrive/Desktop/MLOPS1-dvc-storage`. DagsHub publication is not claimed. The previous DagsHub URL remains configured but is not the default.
7. A fresh Git clone contains the pointer, not data. Configure an accessible DVC remote, then `uv run dvc pull`. This laptop-local remote must be copied or replaced for another machine to access it.
8. Checking out a genuinely older pointer and running `dvc checkout` restores that pointer's dataset layout: newly added processed folders disappear if the old pointer did not contain them. Return with `git checkout main` and `dvc checkout`. Do not do this over uncommitted dataset work. The fresh-clone restoration check is documented in verification.md.

## Dataset corrections

The actual source dataset has 16,643 images: training 9,866, validation 3,430, evaluation 3,347. The previous repository pointer tracked only 18 files. The raw files are unchanged. Deterministic preprocessing produces RGB 128×128 ImageFolder directories, with up to 100 images per class per split in mini (rather than two).

611 duplicate processed pixel images across splits were excluded, keeping evaluation first, validation second, training last. Full processed counts are training 9,566, validation 3,119, evaluation 3,347 (16,032 total). Mini has training 1,100, validation 1,096, evaluation 1,096 (3,292 total). Rice has 96 available unique validation/evaluation images, which satisfies both the <=100 mini requirement and the >=50 contract. See data_preparation.json for the exact class counts and excluded raw paths.
