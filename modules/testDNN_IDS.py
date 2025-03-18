#!/usr/bin/env python
# coding=UTF-8
"""
基于深度神经网络的流量异常检测（分类任务）完整版
包含可视化修复和系统兼容性增强
"""

#region [1]环境配置
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from common_utils import *

# 配置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 字体配置
configure_matplotlib()
import matplotlib.pyplot as plt
#endregion


#region [2]执行数据准备
data_feature, data_label, n_classes, class_names = load_and_preprocess_data()


#数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    data_feature, data_label,
    test_size=0.2,      # 随机选择20%作为测试集
    stratify=data_label,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# 数据预处理
def df_to_dataset(features, labels, shuffle=True, batch_size=32):
    """创建TensorFlow数据管道"""
    # 将特征和标签转换为TensorFlow Dataset对象
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle: # 数据打乱：仅在训练时启用
        dataset = dataset.shuffle(buffer_size=len(features))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

BATCH_SIZE = 64
train_ds = df_to_dataset(X_train, y_train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(X_val, y_val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(X_test, y_test, shuffle=False, batch_size=BATCH_SIZE)
#endregion

#region [3]模型构建与训练
# 构建模型
def build_dnn_model():
    """构建深度神经网络"""
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train.values)
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(20, activation='selu', kernel_initializer='lecun_normal'),
        layers.Dropout(0.2),
        layers.Dense(20, activation='selu', kernel_initializer='lecun_normal'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

model = build_dnn_model()

# 模型训练
def compile_and_train(model):
    """编译与训练模型"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop],
        verbose=1
    )
    return history

print("\n开始训练模型...")
history = compile_and_train(model)
#endregion

#region [4]执行可视化
plot_training_history(history, 'DNN')

# 生成预测结果
print("\n生成预测结果...")
y_pred = model.predict(test_ds).argmax(axis=1)
y_true = y_test.values

# 混淆矩阵可视化
plot_confusion_matrix(y_true, y_pred, class_names, 'DNN')
save_model_and_labels(model, class_names, 'DNN')

# 最终评估
test_loss, test_acc = model.evaluate(test_ds, verbose=0)    # metrics=['accuracy']返回[loss, accuracy]
print(f"\n测试集准确率: {test_acc:.4f}")
print(f"测试集损失值: {test_loss:.4f}")


