# PowerShell helper to initialize DVC and configure a local remote
Write-Host "Installing dvc (if not present) and initializing repository..."
python -m pip install --upgrade pip
python -m pip install dvc

if (-not (Test-Path -Path "./.dvc")) {
    dvc init
} else {
    Write-Host "DVC already initialized."
}

if (-not (Test-Path -Path "./data_storage")) {
    New-Item -ItemType Directory -Path ./data_storage | Out-Null
}

dvc remote add -d localstorage ./data_storage
Write-Host "Added DVC remote 'localstorage' -> ./data_storage"

Write-Host "Done. Use 'dvc add <path>' to track files and 'dvc push' to push to the local remote." 