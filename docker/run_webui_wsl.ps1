param(
    [string]$Distro = "Ubuntu",
    [string]$ImageName = "indextts2:cu128-ds",
    [string]$WorkRoot = "/root/indextts2-docker",
    [int]$Port = 7860,
    [string]$ContainerName = "indextts2-webui",
    [string[]]$WebuiArgs = @()
)

$ErrorActionPreference = "Stop"

$scriptPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "run_webui_container.sh"))
if ($scriptPath -notmatch "^([A-Za-z]):\\(.*)$") {
    throw "Cannot convert script path to WSL path: $scriptPath"
}
$drive = $Matches[1].ToLowerInvariant()
$rest = $Matches[2] -replace "\\", "/"
$scriptPathWsl = "/mnt/$drive/$rest"
$args = @(
    "-d", $Distro,
    "-u", "root",
    "--",
    "bash",
    $scriptPathWsl,
    $ImageName,
    $WorkRoot,
    "$Port",
    $ContainerName
)
foreach ($arg in $WebuiArgs) {
    $args += $arg
}

Start-Process -FilePath "wsl.exe" -ArgumentList $args -WindowStyle Hidden

Write-Host "Started IndexTTS2 WebUI launcher for WSL distro '$Distro'."
Write-Host "Container: $ContainerName"
Write-Host "URL: http://127.0.0.1:$Port/"
Write-Host "Logs: wsl -d $Distro -u root -- docker logs -f $ContainerName"
