import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
from sklearn.metrics import roc_curve, auc
import pickle

def load_data(datafile, labelfile):
    data = pd.read_csv(datafile)
    label = pd.read_csv(labelfile)
    feature_data = data.iloc[:, :]
    label_data = label.iloc[:, :]
    label_data = label_data.values.ravel()
    return feature_data, label_data

def lightgbm_output(feature_data, label_data):
    # 数据划分：训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        feature_data, label_data, test_size=0.3, random_state=33
    )
    # 创建 LightGBM 数据集
    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val)

    # 设置参数
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'binary',  # 二分类任务
        'metric': 'binary_logloss',
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'num_leaves': 55,
        'min_data': 35,
        'max_depth': 6,
        'bagging_seed': 33,
        'feature_fraction_seed': 33,
        'data_random_seed': 33,
    }

    # 训练参数
    callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=33)]

    # 训练模型
    clf = lgb.train(params, d_train, num_boost_round=300, valid_sets=[d_val], callbacks=callbacks)

    # 保存模型到文件
    with open('lightgbm_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    # 对训练集和验证集进行预测
    train_pred_prob = clf.predict(X_train)  # 训练集预测概率
    val_pred_prob = clf.predict(X_val)  # 验证集预测概率

    # 将概率转为类别标签（通过阈值0.5进行分类）
    train_pred = (train_pred_prob >= 0.5).astype(int)     # 调整此处可增加似然比数值
    val_pred = (val_pred_prob >= 0.5).astype(int)

    # 计算验证集评估指标
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_pred_prob)

    # 输出评估指标
    print('LightGBM classification model evaluation on validation set:')
    print(f'accuracy: {accuracy:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall: {recall:.5f}')
    print(f'f1_score: {f1:.5f}')
    print(f'auc: {auc:.5f}')

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(y_val, val_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'LightGBM (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    datafile = 'mimic_dataset/x_mimiciv_atrial_fibrillation_first_full_imputed_del_shap_7.csv'
    labelfile = 'mimic_dataset/y_mimiciv_atrial_fibrillation_first_full_imputed.csv'

    feature_data, label_data = load_data(datafile, labelfile)

    lightgbm_output(feature_data, label_data)