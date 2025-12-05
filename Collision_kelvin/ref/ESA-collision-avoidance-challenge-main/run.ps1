# PowerShell 脚本：自动切换到正确目录并运行 predict_risk.py

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 切换到脚本目录
Set-Location $ScriptDir

Write-Host "Current directory: $PWD" -ForegroundColor Green
Write-Host ""
Write-Host "Running predict_risk.py..." -ForegroundColor Cyan
Write-Host ""

# 运行 Python 脚本
python predict_risk.py

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
