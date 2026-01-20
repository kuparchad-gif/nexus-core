# scripts\run_with_watchdog.ps1
.\install.ps1

$jobs = @()
$services = "aegis","psyshield","memory_sentinel"

foreach ($s in $services) {
    $job = Start-Job -ScriptBlock {
        param($name)
        & ".\run_$name.ps1"
    } -ArgumentList $s
    $jobs += $job
}

Start-Sleep -Seconds 3
& ".\local_watcher.ps1" -AutoRestart -AlertOnFail

# Cleanup on exit
Read-Host "Press ENTER to stop all..."
$jobs | Stop-Job
$jobs | Remove-Job