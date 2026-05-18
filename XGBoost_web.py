# -------------------------------------------------
# 2. XGBoost（patient-level split）
# -------------------------------------------------
def xgboost_output(feature_data, label_data):
    # -----------------------------
    # 2.1 基于 patient_id 划分
    # -----------------------------
    patient_ids = feature_data.iloc[:, 0]   # patient / subject / stay ID
    X_features = feature_data.iloc[:, 1:]   # 真正用于建模的特征

    # =================================================
    # 🚨 核心防错保护：强制统一特征名和顺序，与前端完全一致！
    # =================================================
    FRONTEND_FEATURES = [
        "Delta WBC",
        "Mean Urine Output",
        "Delta eGFR",
        "Delta Bicarbonate",
        "Max BUN",
        "Ventilation",
        "Diuretics",
        "Age",
        "Weight",
        "SOFA"
    ]
    
    try:
        # 强制按照前端所需的列名和顺序筛选数据
        X_features = X_features[FRONTEND_FEATURES]
    except KeyError as e:
        # 如果报错，说明你 CSV 里的表头名字跟前端不一样（比如大小写不对，或者多/少了空格）
        print(f"\n❌ 严重错误: CSV 文件中找不到对应的特征列！")
        print(f"缺失的列名是: {e}")
        print("请检查你的 CSV 表头是否与 FRONTEND_FEATURES 里的拼写和大小写完全一致！")
        return  # 直接终止程序，不生成错误模型

    unique_patients = patient_ids.unique()

    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=0.3,
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
    # 2.3 参数设置
    # -----------------------------
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,  
        'max_depth': 7,  
        'lambda': 3,  
        'alpha': 0.6,  
        'subsample': 0.8,  
        'colsample_bytree': 0.8,  
        'min_child_weight': 3,  
        'eta': 0.005,  
        'seed': 42,
        'nthread': 5,
        'eval_metric': 'auc',
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1) 
    }

    # -----------------------------
    # 2.4 模型训练
    # -----------------------------
    evals = [(d_train, 'train'), (d_val, 'val')]

    clf = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=42,
        verbose_eval=10
    )

    # -----------------------------
    # 2.5 保存模型
    # -----------------------------
    # 🚨 修复：保存的文件名必须和网页读取的名字完全一样！
    model_filename = 'xgboost_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\n✅ 模型已成功保存为: {model_filename}，请将此文件与 streamlit_app.py 放在同一目录下。")

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
