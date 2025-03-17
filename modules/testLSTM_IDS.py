#!/usr/bin/env python
# coding=UTF-8
"""
基于LSTM的流量异常检测系统（集成标准化层版本）
"""

# %% [1] 环境配置
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 配置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 跨平台字体配置
def configure_matplotlib():
    """解决中文显示问题的终极方案"""
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 定义各平台字体路径
    FONT_PATHS = {
        'win': [
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simhei.ttc'   # 黑体
        ],
        'linux': [
            '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        ],
        'darwin': [
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/Library/Fonts/Microsoft/SimHei.ttf'
        ]
    }

    # 自动检测操作系统
    platform = 'win' if os.name == 'nt' else 'linux' if sys.platform.startswith('linux') else 'darwin'

    # 尝试加载所有可用字体
    success = False
    for font_path in FONT_PATHS[platform]:
        try:
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
            success = True
            break
        except Exception as e:
            continue

    if not success:
        print("警告：未能加载中文字体，将使用默认字体")
        plt.rcParams['font.family'] = 'sans-serif'

    plt.rcParams['axes.unicode_minus'] = False

configure_matplotlib()
import matplotlib.pyplot as plt

# %% [2] 数据准备
def get_absolute_path(relative_path):
    """获取跨平台绝对路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, '..', *relative_path.split('/'))

def load_and_preprocess_data():
    """数据加载与预处理流程"""
    csv_path = get_absolute_path('category/binary_classification.csv')
    print(f"数据文件路径验证: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件未找到: {csv_path}")

    df = pd.read_csv(csv_path)

    # 标签编码
    label_categories = pd.Categorical(df['Label'])
    global n_classes, class_names
    class_names = label_categories.categories.tolist()
    n_classes = len(class_names)
    df['Label'] = label_categories.codes

    # 特征工程配置
    features = [
        'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
        'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
        'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
        'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
        'Average_Packet_Size'
    ]
    df[features] = df[features].astype('float32')

    return df[features], df['Label']

# 执行数据准备
features, labels = load_and_preprocess_data()
labels = labels.to_frame(name='Label')

# %% [3] 时序数据生成器
PAST_HISTORY = 1000  # 历史窗口
FUTURE_TARGET = 10   # 预测窗口
STEP = 6             # 采样间隔

def create_time_series(features, labels, start_idx, end_idx, hist_size, pred_size, step):
    """生成时序数据集"""
    data = []
    labels_out = []

    end_idx = end_idx or (len(features) - pred_size)
    start_idx = max(start_idx, hist_size)

    for i in range(start_idx, end_idx):
        indices = range(i-hist_size, i, step)
        data.append(features.iloc[indices].values)
        labels_out.append(labels.iloc[i+pred_size].values)

    return np.array(data), np.array(labels_out)

# 分割数据集
TRAIN_SPLIT = 30000
train_features, train_labels = features.iloc[:TRAIN_SPLIT], labels.iloc[:TRAIN_SPLIT]
val_features, val_labels = features.iloc[TRAIN_SPLIT:], labels.iloc[TRAIN_SPLIT:]

# 生成时序数据
x_train, y_train = create_time_series(
    train_features, train_labels,
    start_idx=0, end_idx=None,
    hist_size=PAST_HISTORY,
    pred_size=FUTURE_TARGET,
    step=STEP
)

x_val, y_val = create_time_series(
    val_features, val_labels,
    start_idx=PAST_HISTORY,
    end_idx=None,
    hist_size=PAST_HISTORY,
    pred_size=FUTURE_TARGET,
    step=STEP
)

# %% [4] 模型构建与训练
def build_lstm_model(input_shape):
    """构建LSTM模型"""
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(train_features.values)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        normalization_layer,
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label='ovr')]
    )
    return model

model = build_lstm_model((x_train.shape[1], x_train.shape[2]))

# 训练配置
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=3,
    mode='max',
    restore_best_weights=True
)

BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(1)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

# 训练模型
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stopping],
    verbose=1
)

# %% [5] 可视化模块
def ensure_dir(path):
    """确保目录存在且有写入权限"""
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.access(path, os.W_OK):
        raise PermissionError(f"无权限写入目录: {path}")

def plot_training_history(history):
    """可视化训练过程指标"""
    plt.figure(figsize=(12, 5), dpi=150)

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练损失变化', fontsize=12, pad=15)
    plt.ylabel('损失值', fontsize=10)
    plt.xlabel('训练轮次', fontsize=10)
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('训练准确率变化', fontsize=12, pad=15)
    plt.ylabel('准确率', fontsize=10)
    plt.xlabel('训练轮次', fontsize=10)
    plt.legend()

    # 保存图表
    output_dir = get_absolute_path('results/figures')
    ensure_dir(output_dir)
    plt.savefig(
        os.path.join(output_dir, 'lstm_training_metrics.png'),
        bbox_inches='tight',
        pad_inches=0.2
    )
    plt.close()
    print(f"训练指标图表已保存至: {output_dir}")

def plot_confusion_matrix(y_true, y_pred, classes):
    """生成并保存混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)

    plt.title('LSTM分类结果混淆矩阵', fontsize=14, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    output_dir = get_absolute_path('results/figures')
    ensure_dir(output_dir)
    plt.savefig(
        os.path.join(output_dir, 'lstm_confusion_matrix.png'),
        bbox_inches='tight',
        pad_inches=0.3
    )
    plt.close()
    print(f"混淆矩阵已保存至: {output_dir}")

# 执行可视化
if history is not None:
    plot_training_history(history)
else:
    print("警告：无训练历史数据")

# 生成预测结果
print("\n生成预测结果...")
y_pred = model.predict(val_dataset).argmax(axis=1)
y_true = np.concatenate([y for x, y in val_dataset], axis=0)

plot_confusion_matrix(y_true, y_pred, class_names)

# %% [6] 模型保存与评估
def save_model_and_labels(model, class_names):
    """保存模型和标签映射"""
    model_dir = get_absolute_path('models')
    ensure_dir(model_dir)

    # 保存模型
    model_path = os.path.join(model_dir, 'lstm_traffic_model.keras')
    model.save(model_path)

    # 保存标签映射
    label_map = {str(i): name for i, name in enumerate(class_names)}
    label_path = os.path.join(model_dir, 'label_mapping.json')
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\n模型已保存至: {model_path}")
    print(f"标签映射已保存至: {label_path}")

save_model_and_labels(model, class_names)

# 最终评估
test_loss, test_acc, test_auc = model.evaluate(val_dataset, verbose=0)
print(f"\n验证集准确率: {test_acc:.4f}")
print(f"验证集AUC: {test_auc:.4f}")
print(f"验证集损失值: {test_loss:.4f}")
