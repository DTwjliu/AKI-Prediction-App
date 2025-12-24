import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import pickle


def load_data(datafile, labelfile):
    data = pd.read_csv(datafile)
    label = pd.read_csv(labelfile)
    feature_data = data.iloc[:, :]
    label_data = label.iloc[:, :]
    label_data = label_data.values.ravel()
    return feature_data, label_data


def xgboost_output(feature_data, label_data):
    # -----------------------------
    # 1. 划分训练集和验证集
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        feature_data, label_data, test_size=0.3, random_state=33
    )

    # -----------------------------
    # 2. 构造 DMatrix
    # -----------------------------
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_val, label=y_val)

    # -----------------------------
    # 3. 参数设置（使用你给定的参数）
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
        # 类别不平衡处理
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
    }

    # -----------------------------
    # 4. 模型训练
    # -----------------------------
    evals = [(d_train, 'train'), (d_val, 'val')]

    clf = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=33,
        verbose_eval=1
    )

    # -----------------------------
    # 5. 保存模型
    # -----------------------------
    with open('xgboost_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    # -----------------------------
    # 6. 预测
    # -----------------------------
    train_pred_prob = clf.predict(d_train)
    val_pred_prob = clf.predict(d_val)

    train_pred = (train_pred_prob >= 0.5).astype(int)
    val_pred = (val_pred_prob >= 0.5).astype(int)

    # -----------------------------
    # 7. 验证集评估指标
    # -----------------------------
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    auc_score = roc_auc_score(y_val, val_pred_prob)

    print('XGBoost classification model evaluation on validation set:')
    print(f'accuracy:  {accuracy:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall:    {recall:.5f}')
    print(f'f1_score:  {f1:.5f}')
    print(f'auc:       {auc_score:.5f}')

    # -----------------------------
    # 8. 绘制 ROC 曲线
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


if __name__ == '__main__':
    datafile = 'mimic_dataset/111_reduce.csv'
    labelfile = 'mimic_dataset/222.csv'

    feature_data, label_data = load_data(datafile, labelfile)
    xgboost_output(feature_data, label_data)
