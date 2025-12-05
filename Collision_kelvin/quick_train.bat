@echo off
REM 快速训练脚本 - 推荐配置

echo ========================================
echo 卫星碰撞风险预测 - 快速训练
echo ========================================

set PYTHON=D:\ProgramData\anaconda3\envs\sg2ada\python.exe

echo.
echo 选择配置:
echo [1] 稳定配置 (推荐) - 降低学习率 + 增大批次
echo [2] 强正则化 - 更大的 dropout
echo [3] 小模型 - 减少参数量
echo [4] RankLoss 优化 - 调整权重
echo [5] 完整优化 - LDS + RankLoss + SupCR
echo [6] 自定义
echo.

set /p choice="请输入选项 (1-6): "

if "%choice%"=="1" goto config1
if "%choice%"=="2" goto config2
if "%choice%"=="3" goto config3
if "%choice%"=="4" goto config4
if "%choice%"=="5" goto config5
if "%choice%"=="6" goto config6

echo 无效选项！
pause
exit

:config1
echo.
echo [配置1] 稳定配置
echo - hidden_size: 96
echo - num_layers: 3
echo - dropout: 0.4
echo - batch_size: 128
echo - lr: 0.00005
echo - lambda_rank: 0.1
echo.
%PYTHON% main.py --model LSTM --hidden_size 96 --num_layers 3 --dropout 0.4 --batch_size 128 --lr 0.00005 --lambda_rank 0.1 --epoch 50
goto end

:config2
echo.
echo [配置2] 强正则化
echo - hidden_size: 96
echo - num_layers: 3
echo - dropout: 0.5
echo - lambda_rank: 0.1
echo.
%PYTHON% main.py --model LSTM --hidden_size 96 --num_layers 3 --dropout 0.5 --lambda_rank 0.1 --epoch 50
goto end

:config3
echo.
echo [配置3] 小模型
echo - hidden_size: 64
echo - num_layers: 2
echo - dropout: 0.5
echo - lambda_rank: 0.1
echo.
%PYTHON% main.py --model LSTM --hidden_size 64 --num_layers 2 --dropout 0.5 --lambda_rank 0.1 --epoch 50
goto end

:config4
echo.
echo [配置4] RankLoss 优化
echo - lambda_rank: 0.15
echo - rank_margin: 0.5
echo.
%PYTHON% main.py --model LSTM --lambda_rank 0.15 --rank_margin 0.5 --epoch 50
goto end

:config5
echo.
echo [配置5] 完整优化
echo - lambda_rank: 0.1
echo - use_supcr: True
echo - lambda_sup: 0.3
echo.
%PYTHON% main.py --model LSTM --lambda_rank 0.1 --use_supcr --lambda_sup 0.3 --epoch 50
goto end

:config6
echo.
echo 请直接运行: python main.py --model LSTM [参数]
echo 查看 OPTIMIZATION_GUIDE.md 了解所有参数
pause
exit

:end
echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 查看结果:
echo   python analyze_results.py
echo.
pause
