@echo off
REM 快速 Git 提交和推送脚本

echo ========================================
echo Git 快速上传
echo ========================================

REM 1. 查看状态
echo.
echo [1/4] 检查修改...
git status

REM 2. 添加所有修改
echo.
echo [2/4] 添加文件...
git add .

REM 3. 提交
echo.
set /p commit_msg="[3/4] 输入提交信息: "
if "%commit_msg%"=="" set commit_msg="update"
git commit -m "%commit_msg%"

REM 4. 推送
echo.
echo [4/4] 推送到 GitHub...
git push

echo.
echo ========================================
echo ✅ 上传完成！
echo ========================================
pause
