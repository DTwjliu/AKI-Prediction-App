import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import pickle


# -------------------------------------------------
# 1. 数据加载
# -------------------------------------------------
def load_data(datafile, labelfile):
    data = pd.read_csv(datafile)
    label = pd.read_csv(labelfile)

    feature_data = data.iloc[:, :]      # 第一列是 patient_id
    label_data = label.iloc[:, :].values.ravel()

    return feature_data, label_data


# -------------------------------------------------
# 2. XGBoost（patient-level split）
# -------------------------------------------------
def xgboost_output(feature_data, label_data):
    # -----------------------------
    # 2.1 基于 patient_id 划分
    # -----------------------------
    patient_ids = feature_data.iloc[:, 0]   # patient / subject / stay ID
    X_features = feature_data.iloc[:, 1:]   # 真正用于建模的特征

    unique_patients = patient_ids.unique()

    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        random_state=42
    )

    train_mask = patient_ids.isin(train_patients)
    val_mask = patient_ids.isin(val_patients)

    X_train = X_features.loc[train_mask]
    X_val   = X_features.loc[val_mask]
    y_train = label_data[train_mask]
    y_val   = label_data[val_mask]

    print(f'Train patients: {len(train_patients)}')
    print(f'Val patients:   {len(val_patients)}')
    print(f'Train samples:  {X_train.shape[0]}')
    print(f'Val samples:    {X_val.shape[0]}')

    # -----------------------------
    # 2.2 构造 DMatrix
    # -----------------------------
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_val, label=y_val)

    # -----------------------------
    # 2.3 参数设置（你的版本）
    # -----------------------------
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.01,
        'max_depth': 6,
        'lambda': 1.0,
        'alpha': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'eta': 0.02,
        'seed': 42,
        'nthread': 3,
        'eval_metric': 'auc',
        # 类别不平衡处理（仅基于训练集）
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
    }

    # -----------------------------
    # 2.4 模型训练
    # -----------------------------
    evals = [(d_train, 'train'), (d_val, 'val')]

    clf = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=42,
        verbose_eval=1
    )

    # -----------------------------
    # 2.5 保存模型
    # -----------------------------
    with open('xgboost_patient_level.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # -----------------------------
    # 2.6 预测
    # -----------------------------
    val_pred_prob = clf.predict(d_val)
    val_pred = (val_pred_prob >= 0.5).astype(int)

    # -----------------------------
    # 2.7 验证集评估
    # -----------------------------
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    auc_score = roc_auc_score(y_val, val_pred_prob)

    print('\nXGBoost classification model evaluation (patient-level split):')
    print(f'accuracy:  {accuracy:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall:    {recall:.5f}')
    print(f'f1_score:  {f1:.5f}')
    print(f'auc:       {auc_score:.5f}')

    # -----------------------------
    # 2.8 ROC 曲线
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_val, val_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'XGBoost (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.show()


# -------------------------------------------------
# 3. 主程序入口
# -------------------------------------------------
if __name__ == '__main__':
    datafile = 'mimic_dataset/111_reduce.csv'
    labelfile = 'mimic_dataset/222.csv'

    feature_data, label_data = load_data(datafile, labelfile)
    xgboost_output(feature_data, label_data)
