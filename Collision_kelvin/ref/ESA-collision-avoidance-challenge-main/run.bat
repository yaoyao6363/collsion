@echo off
REM 批处理脚本：自动切换到正确目录并运行 predict_risk.py

cd /d "%~dp0"
echo Current directory: %CD%
echo.
echo Running predict_risk.py...
echo.

python predict_risk.py

echo.
echo Done! Press any key to exit...
pause > nul
