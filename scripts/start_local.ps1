# Start the verified local stack without using Docker or exposing network ports.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run uv sync --frozen first.' }
$logsPath = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
$env:PYTHONUTF8 = '1'
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
$env:INFERENCE_URL = 'http://127.0.0.1:8000'
function Start-LocalService($name, $port, $arguments) {
    try {
        $probe = [System.Net.Sockets.TcpClient]::new('127.0.0.1', $port)
        $probe.Dispose()
        Write-Host "$name port $port already has a listener; leaving it unchanged."
        return
    } catch [System.Net.Sockets.SocketException] { }

    Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logsPath "$name.log") -RedirectStandardError (Join-Path $logsPath "$name-error.log")
    Write-Host "Started $name on localhost:$port"
}
function Wait-LocalHealth($url) {
    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Seconds 1
    }
    throw "Service did not become healthy: $url. Check the logs directory."
}
Start-LocalService 'mlflow' 5000 @('-m','mlflow','server','--host','127.0.0.1','--port','5000','--workers','1','--backend-store-uri','sqlite:///mlflow.db','--serve-artifacts','--artifacts-destination','./mlruns')
Wait-LocalHealth 'http://127.0.0.1:5000/health'
Start-LocalService 'inference' 8000 @('-m','uvicorn','src.food11.serve:app','--host','127.0.0.1','--port','8000')
Wait-LocalHealth 'http://127.0.0.1:8000/health'
# 8501 is already used by another project on this laptop.
Start-LocalService 'frontend' 8502 @('-m','streamlit','run','frontend/app.py','--server.address','127.0.0.1','--server.port','8502','--browser.gatherUsageStats','false')
Wait-LocalHealth 'http://127.0.0.1:8502/_stcore/health'
