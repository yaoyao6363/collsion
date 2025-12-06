@echo off
REM ============================================================
REM Kessler CDM数据生成与风险预测快速启动脚本 (Windows)
REM ============================================================

echo ============================================================
echo Kessler CDM风险预测 - 快速启动
echo ============================================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

echo [1/5] 检查依赖...
pip show kessler >nul 2>&1
if errorlevel 1 (
    echo Kessler未安装，正在安装...
    cd kessler
    pip install -e .
    cd ..
)

echo.
echo [2/5] 生成CDM数据 (从Kelvins数据集)...
python generate_cdm_data.py --mode kelvins --input_csv ./dataset/train_data.csv --output_dir ./processed_cdms/ --num_events 1000

if errorlevel 1 (
    echo [错误] CDM数据生成失败
    pause
    exit /b 1
)

echo.
echo [3/5] 预处理数据...
python preprocess_cdm_data.py --input_csv ./processed_cdms/kelvins_cdms.csv --output_csv ./dataset/processed_train_data.csv

if errorlevel 1 (
    echo [错误] 数据预处理失败
    pause
    exit /b 1
)

echo.
echo [4/5] 训练模型 (TRANSFORMER)...
python main.py --model TRANSFORMER --epoch 50 --batch_size 256 --lr 0.0005 --data_path ./dataset/ --dataset_train processed_train_data.csv

if errorlevel 1 (
    echo [错误] 模型训练失败
    pause
    exit /b 1
)

echo.
echo [5/5] 测试模型...
python main.py --model TRANSFORMER --only_test --data_path ./dataset/ --dataset_test processed_train_data.csv

echo.
echo ============================================================
echo 完成! 结果保存在 ./results/ 目录
echo ============================================================
pause
