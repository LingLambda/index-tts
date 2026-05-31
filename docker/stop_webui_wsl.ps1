param(
    [string]$Distro = "Ubuntu",
    [string]$ContainerName = "indextts2-webui"
)

$ErrorActionPreference = "Stop"

wsl -d $Distro -u root -- docker rm -f $ContainerName
Write-Host "Stopped IndexTTS2 WebUI container '$ContainerName' in WSL distro '$Distro'."
