# Deepfake Shield — Backend Launcher (Windows Priority & QuickEdit Shield)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Deepfake Shield — Real-Time Inference Backend" -ForegroundColor Cyan
Write-Host "  Process Priority: AboveNormal | EcoQoS: Disabled" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

# Disable QuickEdit Mode in PowerShell host to prevent accidental freeze on click
try {
    $Kernel32 = Add-Type -MemberDefinition @"
        [DllImport("kernel32.dll")]
        public static extern IntPtr GetStdHandle(int nStdHandle);
        [DllImport("kernel32.dll")]
        public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
        [DllImport("kernel32.dll")]
        public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
"@ -Name "WinConsoleUtil" -Namespace "DeepfakeShield" -PassThru -ErrorAction SilentlyContinue

    $hStdin = $Kernel32::GetStdHandle(-10)
    $mode = 0
    if ($Kernel32::GetConsoleMode($hStdin, [ref]$mode)) {
        $newMode = ($mode -band -bnot 0x0040) -bor 0x0080
        $Kernel32::SetConsoleMode($hStdin, $newMode) | Out-Null
        Write-Host "  [OK] Console QuickEdit disabled (terminal click will not pause inference)" -ForegroundColor DarkGray
    }
} catch {
    # Ignore if non-console
}

# Start uvicorn process
$pyPath = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pyPath)) {
    $pyPath = "python"
}

$Process = Start-Process -FilePath $pyPath -ArgumentList "-m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000" -PassThru
try {
    $Process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
    Write-Host "  [OK] Process priority elevated to AboveNormal (PID: $($Process.Id))" -ForegroundColor Green
} catch {
    Write-Host "  [!] Could not elevate process priority: $_" -ForegroundColor Yellow
}

Write-Host "`nServer listening at: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "Press Ctrl+C to stop backend.`n" -ForegroundColor DarkGray
$Process.WaitForExit()
