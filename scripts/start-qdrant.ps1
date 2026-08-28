[CmdletBinding()]
param(
    [string]$Image = 'qdrant/qdrant:v1.19.0',
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$containerName = 'voiceloop-qdrant'
$volumeName = 'voiceloop-qdrant-data'
$restartPolicy = 'unless-stopped'

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Test-Qdrant {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:6333/healthz' -TimeoutSec 2 |
            Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Test-Docker {
    cmd.exe /c 'docker info >nul 2>nul'
    return $LASTEXITCODE -eq 0
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
    docker update --restart $restartPolicy $containerName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Nie udalo sie ustawic automatycznego restartu Qdranta.'
    }
    $running = docker inspect --format '{{.State.Running}}' $containerName
    if ($Restart) {
        docker restart $containerName | Out-Null
    }
    elseif ($running -ne 'true') {
        if (Test-Port 6333) {
            throw "Port 6333 jest zajety, ale kontener $containerName nie dziala."
        }
        docker start $containerName | Out-Null
    }
    elseif (-not (Test-Qdrant)) {
        docker restart $containerName | Out-Null
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Nie udalo sie uruchomic kontenera Qdranta.'
    }
}
else {
    if (Test-Port 6333) {
        throw (
            'Port 6333 jest zajety przez proces spoza zarzadzanego kontenera ' +
            "$containerName."
        )
    }
    $volume = docker volume ls --filter "name=^$volumeName$" --format '{{.Name}}'
    if ($volume -ne $volumeName) {
        docker volume create $volumeName | Out-Null
    }
    docker run --detach `
        --name $containerName `
        --restart $restartPolicy `
        --stop-timeout 30 `
        --health-cmd "bash -ec 'exec 3<>/dev/tcp/127.0.0.1/6333'" `
        --health-interval 10s `
        --health-timeout 3s `
        --health-retries 6 `
        --health-start-period 15s `
        --log-opt 'max-size=10m' `
        --log-opt 'max-file=3' `
        --publish '127.0.0.1:6333:6333' `
        --publish '127.0.0.1:6334:6334' `
        --volume "${volumeName}:/qdrant/storage" `
        $Image | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Nie udalo sie utworzyc kontenera Qdranta.'
    }
}

$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline) {
    if ((Test-Port 6333) -and (Test-Qdrant)) {
        $policy = docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' $containerName
        if ($policy -ne $restartPolicy) {
            throw "Qdrant dziala bez wymaganej polityki restartu: $policy."
        }
        Write-Output (
            'Qdrant uruchomiony na http://127.0.0.1:6333 ' +
            "(restart=$restartPolicy, image=$Image)."
        )
        return
    }
    Start-Sleep -Seconds 1
}

throw 'Qdrant nie odpowiedzial w ciagu 2 minut.'
