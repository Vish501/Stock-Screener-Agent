# ========================================
# load_environment.ps1
# Load .env and activate project venv
# ========================================

# Path to .env file
# Assumes the path in project root
$envFile = ".env"

if (Test-Path $envFile) {
    Write-Host "Loading environment variables from $envFile..."
    
    # Read all non-empty, non-comment lines
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not [string]::IsNullOrWhiteSpace($line) -and -not $line.StartsWith('#')) {
            $parts = $line -split '=', 2
            if ($parts.Length -eq 2) {
                $name = $parts[0].Trim()
                $value = $parts[1].Trim()

                # Use Set-Item for dynamic env variable
                Set-Item -Path "Env:$name" -Value $value

                Write-Host "Set $name"
            }
        }
    }

    # ===== Verify GPU-related flags =====
    $gpuFlags = @("OLLAMA_USE_GPU", "OLLAMA_FLASH_ATTENTION")
    foreach ($flag in $gpuFlags) {
        $val = [System.Environment]::GetEnvironmentVariable($flag, "Process")
        if ($val) {
             Write-Host "$flag is set to $val"
        } else {
            Write-Warning "$flag is NOT set"
        }
    }  


} else {
    Write-Warning ".env file not found at $envFile"
}