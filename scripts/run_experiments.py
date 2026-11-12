"""Run the controlled Lab 2 comparison, then Lab 5 candidates."""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()
settings = [(0.01,32), (0.001,32), (0.0001,32), (0.001,64)]
comparison=[]
for lr, batch in settings:
    subprocess.run([sys.executable,"-u","src/food11/train.py","--dataset","mini",
                    "--epochs","5","--lr",str(lr),"--batch-size",str(batch),
                    "--freeze-backbone"], check=True)
    experiment=client.get_experiment_by_name("food11")
    runs=client.search_runs([experiment.experiment_id],filter_string="attributes.status = 'FINISHED'",
                            order_by=["attributes.start_time DESC"],max_results=1)
    comparison.append(runs[0])
best=max(comparison,key=lambda r:r.data.metrics["val_accuracy"])
version=mlflow.register_model(f"runs:/{best.info.run_id}/model","food11")
client.set_registered_model_alias("food11","champion",version.version)
results={"comparison":[{"run_id":r.info.run_id,"params":dict(r.data.params),"metrics":dict(r.data.metrics)} for r in comparison],
         "initial_champion":{"run_id":best.info.run_id,"version":version.version},"candidates":[]}
Path("lab/experiments.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
for lr in [0.000001, float(best.data.params["lr"])]:
    subprocess.run([sys.executable,"-u","src/food11/train.py","--dataset","mini",
                    "--epochs","5","--lr",str(lr),"--batch-size",best.data.params["batch_size"],
                    "--freeze-backbone"],check=True)
    run=client.search_runs([experiment.experiment_id],filter_string="attributes.status = 'FINISHED'",
                           order_by=["attributes.start_time DESC"],max_results=1)[0]
    candidate=mlflow.register_model(f"runs:/{run.info.run_id}/model","food11")
    subprocess.run([sys.executable,"-u","src/food11/promote.py","--version",candidate.version],check=True)
    tags=dict(client.get_model_version("food11",candidate.version).tags)
    results["candidates"].append({"run_id":run.info.run_id,"version":candidate.version,"tags":tags})
    Path("lab/experiments.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
