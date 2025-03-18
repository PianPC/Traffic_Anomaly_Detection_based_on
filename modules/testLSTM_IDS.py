#!/usr/bin/env python
# coding=UTF-8
"""
基于LSTM的流量异常检测系统（集成标准化层版本）
"""

#region [1] 环境配置
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from common_utils import *

# 配置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 减少冗余日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # 禁用可能不稳定的优化



configure_matplotlib()
import matplotlib.pyplot as plt
#endregion


#region [2]执行数据准备
features, labels, n_classes, class_names = load_and_preprocess_data()
labels = labels.to_frame(name='Label')
'''
​**.to_frame() 方法**：
这是 Pandas Series 的内置方法，用于将 Series 转换为 DataFrame
​**name='Label' 参数**：
指定生成的新 DataFrame 的列名
'''
# 时序数据生成器
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
TRAIN_SPLIT = 30000     # 以TRAIN_SPLIT为分界点分成训练集与验证集
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
'''
时间轴: [0............30000............N]
训练集: |-----------|
验证集:             |-------------------|
每个样本: ◼◼◼◼◼...◼ (1000个时间点，间隔6)
预测目标:            ▲ (未来第10点)
'''
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(1)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
#endregion

#region [3] 模型构建与训练
# 构建模型
def build_lstm_model(input_shape):
    """构建LSTM模型"""
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(x_train.reshape(-1, x_train.shape[-1]))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.LSTM(32, return_sequences=True),
    #     tf.keras.layers.Input(shape=input_shape),
    #     normalization_layer,
    #     # tf.keras.layers.LSTM(128, return_sequences=True),
    #     tf.keras.layers.LSTM(64),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Dense(n_classes, activation='softmax')
    # ])
    model = tf.keras.Sequential([
        # 输入层整合到第一个LSTM层
        tf.keras.layers.Input(shape=input_shape),  # 显式定义输入层

        normalization_layer,  # 标准化层位置调整
        tf.keras.layers.LSTM(32),
        # 简化后的结构
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    return model

model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
# input_shape=x_train_single.shape[-2:] 与原项目代码 (x_train.shape[1], x_train.shape[2]) 一个意思

# 训练模型
def compile_and_train(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label='ovr')]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

print("\n开始训练模型...")
history = compile_and_train(model)
#endregion

#region [4]执行可视化
plot_training_history(history, 'LSTM')

# 生成预测结果
print("\n生成预测结果...")
y_pred = model.predict(val_dataset).argmax(axis=1)
y_true = np.concatenate([y for x, y in val_dataset], axis=0)

# 混淆矩阵可视化
plot_confusion_matrix(y_true, y_pred, class_names, 'LSTM')
save_model_and_labels(model, class_names, 'LSTM')

# 最终评估
test_loss, test_acc, test_auc = model.evaluate(val_dataset, verbose=0)  # metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]返回[loss, precision, recall]
print(f"\n验证集准确率: {test_acc:.4f}")
print(f"验证集AUC: {test_auc:.4f}")
print(f"验证集损失值: {test_loss:.4f}")

# 选取单个训练样本
sample_input = x_train[:1]  # 形状 (1, 时间步长, 特征数)

# 定义损失函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# 梯度计算上下文
with tf.GradientTape() as tape:
    # 前向传播计算预测值
    predictions = model(sample_input)
    # 计算损失值
    loss_value = loss_fn(y_train[:1], predictions)

# 计算梯度
grads = tape.gradient(loss_value, model.trainable_variables)

# 分析LSTM层梯度
print("\n梯度范数分析:")
for g, v in zip(grads, model.trainable_variables):
    if 'lstm' in v.name:
        print(f"{v.name:30} {tf.norm(g):.4e}")
