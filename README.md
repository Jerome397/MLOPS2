LAB1:
------------------------------------------------

Question 1: uv init creates the basic Python project structure. pyproject.toml contains project information, Python requirements, and dependencies, .python-version specifies the Python version, README.md is for documentation, and src contains the source code.

Question 2: dvc init creates the .dvc folder and .dvcignore. .dvc/config stores the DVC project configuration, .dvc/.gitignore prevents DVC cache and internal files from being tracked by Git, and .dvcignore tells DVC which files to ignore. The configuration files should be pushed to Git, but cache files and secrets should not.

Question 3: In our setup, the credentials are stored in .dvc/config.local because we used --local. Other options include the normal project configuration, --global, and --system. Credentials should never be pushed to GitHub because they contain sensitive information such as usernames and access tokens.

Question 4: When dvc add data is run, DVC adds the data folder to .gitignore. This prevents Git from tracking the large dataset files because DVC is responsible for versioning the data instead.

Question 5: The data.dvc file contains metadata about the tracked data folder, such as its path, size, number of files, and a hash that identifies the exact version of the dataset. It does not contain the actual image files.

Question 6: The code and DVC metadata files are stored on GitHub, while the actual dataset is not stored there because Git ignores it. The data.dvc file identifies the correct dataset version, and the actual data is stored in the DVC remote on DagsHub.

Question 7: After cloning the Git repository into a new folder, the actual data folder is not present because Git does not track it. The data.dvc file is present, and we use dvc pull to download the dataset from the DVC remote.

Question 8: After checking out an older Git commit and running dvc checkout, the food11_processed and food11_processed_mini folders disappear because the older data.dvc file represents the earlier version of the data before those processed datasets were created.
LAB2:
-----------------------------------------------

Q1: Question 1: pyproject.toml was updated with the new project dependencies such as MLflow, PyTorch, torchvision and scikit-learn. uv.lock was also updated with the exact resolved versions of these packages and their dependencies, allowing the same environment to be reproduced later.

Q2: Question 2: --backend-store-uri sqlite:///mlflow.db tells MLflow where to store tracking metadata such as experiments, runs, parameters and metrics. --default-artifact-root ./mlruns specifies where run artifacts such as trained models and other generated files are stored. Metadata describes the experiment and its results, while artifacts are actual files produced by the run.

Question 3: mlflow.db and mlruns/ should not be stored in Git because they contain local experiment outputs rather than source code and can change frequently or become large. They should not be tracked by DVC either because they are experiment-tracking outputs already managed by MLflow, while DVC is being used to version the project datasets.

Question 4: The first time mlflow.set_experiment("food11") is called, MLflow automatically creates the food11 experiment because it does not already exist. It then makes this experiment active, so new runs are logged under it.

Question 5: mlflow.log_param records a fixed configuration value for the run, such as the learning rate or batch size, while mlflow.log_metric records numerical results produced during or after training, such as loss or accuracy. Metrics use a step because the same metric can change over time, for example at every epoch. Parameters do not need a step because they remain fixed for the entire run.

Question 6: The MLflow UI shows the run parameters, metric charts and the logged model artifact. Because the tracking server was configured with --default-artifact-root ./mlruns, the model artifact is stored locally inside the mlruns directory under the corresponding experiment and run.

Question 7: The best val_accuracy reached 0.73. From the plot, more than one run appears to reach this value, so there is not a single learning rate that is uniquely best based on validation accuracy alone. A higher learning rate is therefore not always better; performance depends on the combination of hyperparameters and the training run.

q8:The parallel coordinates plot shows that different combinations of lr and batch_size can achieve the same high validation accuracy. In your runs, several combinations reached val_accuracy = 0.73, while at least one combination performed much worse. This shows that validation accuracy depends on the interaction between learning rate and batch size rather than simply increasing one parameter.

q9: Question 9: The best run was masked-koi-824
, with a val_accuracy of 0.73. Its Run ID is 71b6f940adcf440083a3ae1dca86ee9b