param(
    [string]$Url = "http://localhost:18080/",
    [string]$HealthUrl = $Url,
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "SilentlyContinue"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)

while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200) {
            Start-Process $Url
            exit 0
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

Write-Error "等待系统启动超时：$Url"
exit 1
