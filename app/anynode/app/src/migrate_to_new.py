# PowerShell Script: Copy System-Specific Files from Original to New Lillith Structure
# Purpose: Recurse C:\CogniKube-Complete-Final, copy essentials (py/js/html/json/yaml/md) to C:\NewLillithModel, skip non-system (caches/logs/duplicates), log to C:\CogniKube-Complete-Final\copy_log.txt.
# Author: Grok (production-grade, July 2025)
# Usage: Run as Admin in C:\. Preserves folders, force creates new.

# Params
$oldRoot = "C:\CogniKube-Complete-Final"  # Original
$newRoot = "C:\NewLillithModel"  # New
$systemExtensions = @(".py", ".js", ".html", ".json", ".yaml", ".md", ".txt", "*__pycache__*",)  # System-specific
$excludePatterns = @("*.log", "*.zip", "*.pyc", ".vs*", ".vscode*", "*backup*", "*Old*")  # Skip non-essentials
$logFile = "C:\copy_log.txt"

# Function: Copy If System-Specific
function Copy-SystemFile {
    param ($source, $destination)
    try {
        New-Item -Path (Split-Path $destination) -ItemType Directory -Force | Out-Null
        Copy-Item -Path $source -Destination $destination -Force
        Add-Content -Path $logFile -Value "Copied: $source to $destination"
    } catch {
        Add-Content -Path $logFile -Value "Error copying $source: $_"
    }
}

# Main Execution
try {
    if (-not (Test-Path $oldRoot)) { Write-Host "Error: Old root missing."; exit }
    
    # Clear/Create log
    if (Test-Path $logFile) { Remove-Item $logFile }
    New-Item -Path $logFile -ItemType File | Out-Null
    
    # Recurse and Copy
    Get-ChildItem -Path $oldRoot -Recurse | ForEach-Object {
        $relPath = $_.FullName.Replace($oldRoot, "")
        $newPath = $newRoot + $relPath
        $isSystem = $systemExtensions -contains $_.Extension -and -not ($excludePatterns | Where-Object { $_.FullName -like $_ })
        if ($isSystem) {
            Copy-SystemFile $_.FullName $newPath
        } else {
            Add-Content -Path $logFile -Value "Skipped: $_.FullName (not system-specific)"
        }
    }
    
    Write-Host "Copy complete. Check $logFile for details."
} catch {
    Write-Host "Error: $_"
}