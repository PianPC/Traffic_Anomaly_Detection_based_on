import os
import pandas as pd
import matplotlib.pyplot as plt

# 跨平台字体配置（必须在其他matplotlib导入前）
def configure_matplotlib():
    """解决中文显示问题的终极方案"""
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 定义各平台字体路径
    FONT_PATHS = {
        'win': [
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simhei.ttc'   # 黑体
        ],
        'linux': [
            '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',  # 文泉驿正黑
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        ],
        'darwin': [
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',  # Mac系统字体
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

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

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

    return df[features], df['Label'], n_classes, class_names

# %% [7] 可视化模块
def ensure_dir(path):
    """确保目录存在且有写入权限"""
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.access(path, os.W_OK):
        raise PermissionError(f"无权限写入目录: {path}")

def plot_training_history(history):
    """绘制训练指标图表"""
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
        os.path.join(output_dir, 'training_metrics.png'),
        bbox_inches='tight',
        pad_inches=0.2
    )
    plt.close()
    print(f"训练指标图表已保存至: {output_dir}")

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title('混淆矩阵', fontsize=14, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    output_dir = get_absolute_path('results/figures')
    ensure_dir(output_dir)
    plt.savefig(
        os.path.join(output_dir, 'confusion_matrix.png'),
        bbox_inches='tight',
        pad_inches=0.3
    )
    plt.close()
    print(f"混淆矩阵已保存至: {output_dir}")

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
