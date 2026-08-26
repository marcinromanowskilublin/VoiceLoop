param(
    [string]$Image = 'qdrant/qdrant:latest',
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$containerName = 'voiceloop-qdrant'
$volumeName = 'voiceloop-qdrant-data'

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Test-Docker {
    cmd.exe /c 'docker info >nul 2>nul'
    return $LASTEXITCODE -eq 0
}

if ((Test-Port 6333) -and -not $Restart) {
    Write-Output 'Qdrant juz dziala.'
    exit 0
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Nie znaleziono Docker CLI.'
}

if (-not (Test-Docker)) {
    $dockerDesktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $dockerDesktop)) {
        throw 'Docker Desktop nie jest uruchomiony ani zainstalowany.'
    }
    Start-Process -FilePath $dockerDesktop
    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline) {
        if (Test-Docker) {
            break
        }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-Docker)) {
        throw 'Docker Desktop nie uruchomil silnika w ciagu 2 minut.'
    }
}

$existing = docker ps -a --filter "name=^/$containerName$" --format '{{.Names}}'
if ($existing -eq $containerName) {
    if ($Restart) {
        docker restart $containerName | Out-Null
    }
    else {
        docker start $containerName | Out-Null
    }
}
else {
    $volume = docker volume ls --filter "name=^$volumeName$" --format '{{.Name}}'
    if ($volume -ne $volumeName) {
        docker volume create $volumeName | Out-Null
    }
    docker run --detach `
        --name $containerName `
        --restart no `
        --publish '127.0.0.1:6333:6333' `
        --publish '127.0.0.1:6334:6334' `
        --volume "${volumeName}:/qdrant/storage" `
        $Image | Out-Null
}

$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline) {
    if (Test-Port 6333) {
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:6333/healthz' -TimeoutSec 2 | Out-Null
            Write-Output 'Qdrant uruchomiony na http://127.0.0.1:6333.'
            exit 0
        }
        catch {
        }
    }
    Start-Sleep -Seconds 1
}

throw 'Qdrant nie odpowiedzial w ciagu 2 minut.'
