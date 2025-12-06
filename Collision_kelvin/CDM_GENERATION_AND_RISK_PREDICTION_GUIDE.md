# Kessler CDM数据生成与碰撞风险预测完整操作手册

## 目录
1. [环境准备](#1-环境准备)
2. [CDM数据生成](#2-cdm数据生成)
3. [数据预处理](#3-数据预处理)
4. [模型训练](#4-模型训练)
5. [风险预测](#5-风险预测)
6. [完整示例代码](#6-完整示例代码)

---

## 1. 环境准备

### 1.1 安装Kessler库

```bash
# 方法1: 使用conda (推荐)
conda install conda-forge::kessler

# 方法2: 使用pip
pip install kessler

# 方法3: 本地安装
cd kessler
pip install -e .
```

### 1.2 安装项目依赖

```bash
pip install torch pandas numpy matplotlib scikit-learn pyro-ppl dsgp4 skyfield
```

### 1.3 验证安装

```python
import kessler
from kessler import Conjunction, CDM, Event, EventDataset
print(f"Kessler version: {kessler.__version__}")
```

---

## 2. CDM数据生成

### 2.1 使用概率模型生成合成CDM数据

Kessler提供了基于Pyro的概率编程模型来生成真实的CDM数据。

#### 方法A: 使用Conjunction类生成单个碰撞事件

```python
import pyro
from kessler import Conjunction

# 初始化碰撞模型
conj = Conjunction(
    time0=58991.90384230018,           # 起始时间 (MJD)
    max_duration_days=7.0,              # 模拟持续时间
    time_resolution=6e5,                # 时间分辨率
    miss_dist_threshold=5e3,            # 碰撞距离阈值 (米)
    mc_samples=100,                     # 蒙特卡洛采样数
    cdm_update_every_hours=8.0,         # CDM更新间隔 (小时)
    collision_threshold=70              # 碰撞阈值 (米)
)

# 生成一个碰撞事件
trace = pyro.infer.Importance(conj.forward, num_samples=1).run()
```

#### 方法B: 批量生成多个事件

创建文件 `generate_cdm_data.py`:

```python
import pyro
import torch
import numpy as np
from kessler import Conjunction, EventDataset
from kessler.util import progress_bar_init, progress_bar_update, progress_bar_end

def generate_synthetic_cdm_dataset(
    num_events=1000,
    output_dir='./synthetic_cdms/',
    save_format='kvn'  # 'kvn' or 'csv'
):
    """
    生成合成CDM数据集
    
    Args:
        num_events: 要生成的事件数量
        output_dir: 输出目录
        save_format: 保存格式 ('kvn' 或 'csv')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化碰撞模型
    conj = Conjunction(
        time0=58991.90384230018,
        max_duration_days=7.0,
        time_resolution=6e5,
        miss_dist_threshold=5e3,
        mc_samples=100,
        cdm_update_every_hours=8.0
    )
    
    events = []
    progress_bar_init('Generating CDM events', num_events, 'Events')
    
    for i in range(num_events):
        progress_bar_update(i)
        
        try:
            # 使用Pyro生成一个碰撞场景
            trace = pyro.infer.Importance(conj.forward, num_samples=1).run()
            
            # 检查是否生成了有效的碰撞
            if 'conj' in trace.nodes and trace.nodes['conj']['value']:
                # 提取生成的CDM列表
                cdms = []
                # 这里需要从trace中提取CDM数据
                # (具体实现取决于Conjunction.forward的返回值)
                
                if cdms:
                    event = Event(cdms=cdms)
                    events.append(event)
                    
                    # 保存单个事件
                    if save_format == 'kvn':
                        for j, cdm in enumerate(cdms):
                            filename = os.path.join(output_dir, f'event{i}_{j}.kvn')
                            cdm.save(filename)
        except Exception as e:
            print(f"Error generating event {i}: {e}")
            continue
    
    progress_bar_end()
    
    # 创建EventDataset
    event_dataset = EventDataset(events=events)
    
    # 保存为CSV格式
    if save_format == 'csv':
        df = event_dataset.to_dataframe()
        df.to_csv(os.path.join(output_dir, 'synthetic_cdms.csv'), index=False)
    
    print(f"Generated {len(events)} events with CDMs")
    return event_dataset

if __name__ == '__main__':
    dataset = generate_synthetic_cdm_dataset(
        num_events=1000,
        output_dir='./synthetic_cdms/',
        save_format='csv'
    )
```

### 2.2 从现有数据生成CDM

如果你已有Kelvins竞赛数据集：

```python
from kessler.data import kelvins_to_event_dataset

# 加载Kelvins数据集并转换为EventDataset
event_dataset = kelvins_to_event_dataset(
    file_name='./dataset/train_data.csv',
    num_events=None,  # None表示加载所有事件
    remove_outliers=True,
    drop_features=['c_rcs_estimate', 't_rcs_estimate']
)

# 保存为CSV
df = event_dataset.to_dataframe()
df.to_csv('./processed_cdms.csv', index=False)
```

---

## 3. 数据预处理

### 3.1 创建数据预处理脚本

创建文件 `preprocess_cdm_data.py`:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_cdm_for_risk_prediction(
    input_csv='./synthetic_cdms.csv',
    output_csv='./processed_data.csv',
    scaler_path='./scaler.pkl'
):
    """
    预处理CDM数据用于风险预测
    """
    # 加载数据
    df = pd.read_csv(input_csv)
    
    # 1. 选择特征
    feature_columns = [
        # 时间特征
        '__CREATION_DATE', '__TCA', '__DAYS_TO_TCA',
        
        # 相对运动特征
        'MISS_DISTANCE', 'RELATIVE_SPEED',
        'RELATIVE_POSITION_R', 'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N',
        'RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T', 'RELATIVE_VELOCITY_N',
        
        # 目标物体状态
        'OBJECT1_X', 'OBJECT1_Y', 'OBJECT1_Z',
        'OBJECT1_X_DOT', 'OBJECT1_Y_DOT', 'OBJECT1_Z_DOT',
        
        # 目标物体协方差 (不确定性)
        'OBJECT1_CR_R', 'OBJECT1_CT_T', 'OBJECT1_CN_N',
        'OBJECT1_CRDOT_RDOT', 'OBJECT1_CTDOT_TDOT', 'OBJECT1_CNDOT_NDOT',
        
        # 追踪物体状态
        'OBJECT2_X', 'OBJECT2_Y', 'OBJECT2_Z',
        'OBJECT2_X_DOT', 'OBJECT2_Y_DOT', 'OBJECT2_Z_DOT',
        
        # 追踪物体协方差
        'OBJECT2_CR_R', 'OBJECT2_CT_T', 'OBJECT2_CN_N',
        'OBJECT2_CRDOT_RDOT', 'OBJECT2_CTDOT_TDOT', 'OBJECT2_CNDOT_NDOT',
    ]
    
    # 目标变量 (如果有)
    target_column = 'COLLISION_PROBABILITY'  # 或 'risk' 等
    
    # 2. 处理缺失值
    df = df.dropna(subset=feature_columns + [target_column])
    
    # 3. 添加事件ID (如果没有)
    if 'event_id' not in df.columns:
        # 基于时间窗口分组
        df['event_id'] = (df.groupby(['OBJECT1_X', 'OBJECT2_X']).ngroup())
    
    # 4. 特征工程
    # 添加派生特征
    df['relative_distance'] = np.sqrt(
        df['RELATIVE_POSITION_R']**2 + 
        df['RELATIVE_POSITION_T']**2 + 
        df['RELATIVE_POSITION_N']**2
    )
    
    df['relative_velocity_magnitude'] = np.sqrt(
        df['RELATIVE_VELOCITY_R']**2 + 
        df['RELATIVE_VELOCITY_T']**2 + 
        df['RELATIVE_VELOCITY_N']**2
    )
    
    # 不确定性度量
    df['object1_uncertainty'] = np.sqrt(
        df['OBJECT1_CR_R'] + df['OBJECT1_CT_T'] + df['OBJECT1_CN_N']
    )
    
    df['object2_uncertainty'] = np.sqrt(
        df['OBJECT2_CR_R'] + df['OBJECT2_CT_T'] + df['OBJECT2_CN_N']
    )
    
    # 更新特征列表
    feature_columns.extend([
        'relative_distance', 
        'relative_velocity_magnitude',
        'object1_uncertainty',
        'object2_uncertainty'
    ])
    
    # 5. 标准化
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # 保存scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 6. 保存处理后的数据
    df.to_csv(output_csv, index=False)
    
    print(f"Preprocessed data saved to {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"Features: {len(feature_columns)}")
    print(f"Events: {df['event_id'].nunique()}")
    
    return df, feature_columns

if __name__ == '__main__':
    df, features = preprocess_cdm_for_risk_prediction()
```

---

## 4. 模型训练

### 4.1 使用现有的训练框架

你的项目已经有完整的训练框架，可以直接使用：

```bash
# 使用LSTM模型
python main.py --model LSTM --epoch 150 --batch_size 256 --lr 0.0005

# 使用Transformer模型
python main.py --model TRANSFORMER --epoch 150 --batch_size 256 --lr 0.0005

# 使用CDEA模型
python main.py --model CDEA --epoch 150 --batch_size 256

# 使用Bayes LSTM模型
python main.py --model BAYES_LSTM --epoch 150 --batch_size 256
```

### 4.2 自定义训练脚本

创建文件 `train_risk_predictor.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from utils.dataset import CDMDataset
from models.lstm import LSTMModel
from models.transformer import TransformerModel

def train_risk_predictor(
    data_path='./processed_data.csv',
    model_type='TRANSFORMER',
    epochs=150,
    batch_size=256,
    learning_rate=0.0005,
    save_path='./results/checkpoint/'
):
    """
    训练碰撞风险预测模型
    """
    # 1. 加载数据
    df = pd.read_csv(data_path)
    
    # 2. 创建数据集
    dataset = CDMDataset(
        data_path=data_path,
        n_latest_cdms=13,  # 每个事件保留最近13个CDM
        is_train=True
    )
    
    # 3. 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 4. 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. 初始化模型
    input_size = dataset.feature_dim
    
    if model_type == 'LSTM':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=3,
            dropout=0.5
        )
    elif model_type == 'TRANSFORMER':
        model = TransformerModel(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.15
        )
    
    # 6. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 7. 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target, mask) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target, mask in val_loader:
                data, target = data.to(device), target.to(device)
                mask = mask.to(device)
                
                output = model(data, mask)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

if __name__ == '__main__':
    model = train_risk_predictor()
```

---

## 5. 风险预测

### 5.1 使用训练好的模型进行预测

创建文件 `predict_risk.py`:

```python
import torch
import pandas as pd
import pickle
from models.transformer import TransformerModel

def predict_collision_risk(
    model_path='./results/checkpoint/best_model.pth',
    data_path='./test_data.csv',
    scaler_path='./scaler.pkl',
    output_path='./predictions.csv'
):
    """
    使用训练好的模型预测碰撞风险
    """
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerModel(
        input_size=64,  # 根据实际特征数调整
        d_model=128,
        nhead=4,
        num_layers=3
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 2. 加载数据
    df = pd.read_csv(data_path)
    
    # 3. 加载scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 4. 预处理
    feature_columns = [col for col in df.columns if col not in ['event_id', 'risk']]
    df[feature_columns] = scaler.transform(df[feature_columns])
    
    # 5. 预测
    predictions = []
    
    with torch.no_grad():
        for event_id in df['event_id'].unique():
            event_data = df[df['event_id'] == event_id][feature_columns].values
            
            # 转换为tensor
            data = torch.FloatTensor(event_data).unsqueeze(0).to(device)
            
            # 预测
            output = model(data)
            risk = output.cpu().numpy()[0, -1, 0]  # 取最后一个时间步的预测
            
            predictions.append({
                'event_id': event_id,
                'predicted_risk': risk
            })
    
    # 6. 保存结果
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    return pred_df

if __name__ == '__main__':
    predictions = predict_collision_risk()
```

---

## 6. 完整示例代码

### 6.1 端到端Pipeline

创建文件 `end_to_end_pipeline.py`:

```python
"""
完整的CDM数据生成到风险预测Pipeline
"""
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_events', type=int, default=1000, help='生成的事件数量')
    parser.add_argument('--model', type=str, default='TRANSFORMER', choices=['LSTM', 'TRANSFORMER', 'CDEA'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    print("=" * 60)
    print("CDM数据生成与碰撞风险预测Pipeline")
    print("=" * 60)
    
    # Step 1: 生成CDM数据
    print("\n[Step 1/5] 生成合成CDM数据...")
    from generate_cdm_data import generate_synthetic_cdm_dataset
    dataset = generate_synthetic_cdm_dataset(
        num_events=args.num_events,
        output_dir='./synthetic_cdms/',
        save_format='csv'
    )
    
    # Step 2: 数据预处理
    print("\n[Step 2/5] 预处理CDM数据...")
    from preprocess_cdm_data import preprocess_cdm_for_risk_prediction
    df, features = preprocess_cdm_for_risk_prediction(
        input_csv='./synthetic_cdms/synthetic_cdms.csv',
        output_csv='./processed_data.csv'
    )
    
    # Step 3: 训练模型
    print(f"\n[Step 3/5] 训练{args.model}模型...")
    os.system(f"python main.py --model {args.model} --epoch {args.epochs} --batch_size {args.batch_size}")
    
    # Step 4: 模型评估
    print("\n[Step 4/5] 评估模型性能...")
    os.system(f"python main.py --model {args.model} --only_test")
    
    # Step 5: 风险预测
    print("\n[Step 5/5] 进行风险预测...")
    from predict_risk import predict_collision_risk
    predictions = predict_collision_risk()
    
    print("\n" + "=" * 60)
    print("Pipeline完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

### 6.2 快速开始脚本

创建文件 `quick_start.sh` (Linux/Mac) 或 `quick_start.bat` (Windows):

```bash
#!/bin/bash
# quick_start.sh

echo "=== Kessler CDM风险预测快速开始 ==="

# 1. 安装依赖
echo "[1/5] 安装依赖..."
pip install -r requirements.txt

# 2. 生成数据
echo "[2/5] 生成CDM数据..."
python generate_cdm_data.py --num_events 1000

# 3. 预处理
echo "[3/5] 预处理数据..."
python preprocess_cdm_data.py

# 4. 训练模型
echo "[4/5] 训练模型..."
python main.py --model TRANSFORMER --epoch 150 --batch_size 256

# 5. 预测
echo "[5/5] 进行预测..."
python predict_risk.py

echo "=== 完成! ==="
```

---

## 7. 常见问题与解决方案

### Q1: 如何调整生成的CDM数据的真实性?

修改`Conjunction`的参数:
- `miss_dist_threshold`: 降低可生成更多接近碰撞的场景
- `mc_samples`: 增加可提高不确定性估计的准确性
- `cdm_update_every_hours`: 调整CDM更新频率

### Q2: 如何处理不平衡的风险数据?

```python
# 在训练时使用加权损失
from torch.nn import MSELoss

# 计算样本权重
weights = compute_sample_weights(targets)
criterion = MSELoss(reduction='none')
loss = (criterion(outputs, targets) * weights).mean()
```

### Q3: 如何提高模型性能?

1. **特征工程**: 添加更多物理意义的特征
2. **模型集成**: 组合多个模型的预测
3. **超参数调优**: 使用网格搜索或贝叶斯优化
4. **数据增强**: 生成更多样化的训练数据

### Q4: 如何实时预测?

```python
# 创建实时预测服务
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    cdm_data = request.json
    risk = model.predict(cdm_data)
    return jsonify({'risk': float(risk)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 8. 性能优化建议

### 8.1 数据生成优化
- 使用多进程并行生成CDM数据
- 缓存中间结果避免重复计算

### 8.2 训练优化
- 使用混合精度训练 (AMP)
- 启用梯度累积处理大batch
- 使用学习率调度器

### 8.3 推理优化
- 模型量化 (INT8)
- 使用ONNX导出加速推理
- 批量预测而非逐个预测

---

## 9. 参考资源

- **Kessler文档**: https://kesslerlib.github.io/kessler/
- **CCSDS CDM标准**: https://public.ccsds.org/Pubs/508x0b1e2c1.pdf
- **相关论文**: 
  - Acciarini et al. (2021) "Kessler: a Machine Learning Library for Spacecraft Collision Avoidance"
  - Pinto et al. (2020) "Towards Automated Satellite Conjunction Management with Bayesian Deep Learning"

---

## 10. 联系与支持

如有问题，请参考:
- GitHub Issues: https://github.com/kesslerlib/kessler/issues
- 邮件: giacomo.acciarini@gmail.com

---

**祝你的碰撞风险预测项目成功！** 🚀
