"""Inspect CI using Git's credential helper without displaying credentials."""
import json
import subprocess
import urllib.request


def api(path, method="GET", body=None):
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, check=True)
    credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    request = urllib.request.Request("https://api.github.com/repos/Jerome397/MLOPS2/" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + credential["password"],
                 "Accept": "application/vnd.github+json"}, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response) if response.status != 204 else None


if __name__ == "__main__":
    result = api("actions/runs?per_page=5")
    print(json.dumps([{k: run[k] for k in ["id", "head_sha", "status", "conclusion", "html_url"]}
                      for run in result["workflow_runs"]], indent=2))
