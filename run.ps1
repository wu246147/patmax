$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $workspace 'build-vs\Release\patmax_gui.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error "GUI executable not found: $exe. Build the project first."
    exit 1
}
$env:PATH = 'D:\Qt\5.15.2\msvc2019_64\bin;D:\opencv4.8.0\opencv\build\x64\vc16\bin;' + $env:PATH
& $exe @args
