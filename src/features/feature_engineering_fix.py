def select_features(self, X, y=None, method=None, n_features=None, threshold=None):
    """
    特征选择
    
    参数:
        X: 特征矩阵
        y: 目标变量（如果使用监督特征选择）
        method: 特征选择方法，可选 'mutual_info', 'f_regression', 'pca', None
        n_features: 选择的特征数量
        threshold: 特征重要性阈值
        
    返回:
        选择后的特征矩阵和特征名称
    """
    if X is None or X.empty:
        logger.warning("无法对空数据进行特征选择")
        return X, []
    
    # 使用配置中的默认值（如果未指定）
    if method is None:
        method = self.feature_selection.get("method", None)
    
    if n_features is None:
        n_features = self.feature_selection.get("n_features", 50)
    
    if threshold is None:
        threshold = self.feature_selection.get("threshold", None)
    
    # 只选择数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    # 如果数值特征少于所需特征数，调整n_features
    if len(numeric_cols) < n_features:
        n_features = len(numeric_cols)
        logger.warning(f"可用特征数量({len(numeric_cols)})少于所需数量({n_features})，已调整为{n_features}")
    
    # 处理缺失值
    if y is not None:
        # 检查目标变量中的NaN
        nan_count = np.isnan(y).sum()
        if nan_count > 0:
            logger.warning(f"目标变量中包含 {nan_count} 个NaN值，将被处理")
            
            # 创建包含特征和目标的DataFrame，以便同时处理NaN
            combined_df = pd.DataFrame(X_numeric)
            combined_df['target'] = y
            
            # 移除包含NaN的行
            combined_df_clean = combined_df.dropna()
            logger.info(f"移除含NaN的行后，剩余 {len(combined_df_clean)} 行数据，原始数据有 {len(combined_df)} 行")
            
            if len(combined_df_clean) == 0:
                logger.error("处理NaN值后无剩余数据，无法进行特征选择")
                return X, []
            
            # 分离特征和目标变量
            X_numeric = combined_df_clean.drop('target', axis=1)
            y = combined_df_clean['target']
    
    # 特征选择
    selected_features = []
    importance_scores = {}
    
    if method == 'mutual_info' and y is not None:
        # 使用互信息选择特征
        selector = SelectKBest(mutual_info_regression, k=n_features)
        X_selected = selector.fit_transform(X_numeric, y)
        
        # 获取选择的特征名称
        mask = selector.get_support()
        selected_features = X_numeric.columns[mask].tolist()
        
        # 记录特征重要性
        scores = selector.scores_
        for i, feature in enumerate(X_numeric.columns):
            importance_scores[feature] = scores[i]
    
    elif method == 'f_regression' and y is not None:
        # 使用F检验选择特征
        selector = SelectKBest(f_regression, k=n_features)
        X_selected = selector.fit_transform(X_numeric, y)
        
        # 获取选择的特征名称
        mask = selector.get_support()
        selected_features = X_numeric.columns[mask].tolist()
        
        # 记录特征重要性
        scores = selector.scores_
        for i, feature in enumerate(X_numeric.columns):
            importance_scores[feature] = scores[i]
    
    elif method == 'pca':
        # 使用PCA降维
        pca = PCA(n_components=n_features)
        X_selected = pca.fit_transform(X_numeric)
        
        # PCA没有选择原始特征，而是创建新的组件
        selected_features = [f'PC{i+1}' for i in range(n_features)]
        
        # 记录特征重要性（使用解释方差比例）
        explained_variance = pca.explained_variance_ratio_
        for i, feature in enumerate(selected_features):
            importance_scores[feature] = explained_variance[i]
            
        # 返回PCA转换后的数据作为DataFrame
        X_selected_df = pd.DataFrame(X_selected, index=X_numeric.index, columns=selected_features)
        return X_selected_df, selected_features
    
    else:
        # 不进行特征选择，保留所有数值特征
        X_selected = X_numeric.values
        selected_features = numeric_cols.tolist()
        
        # 默认的特征重要性为1
        for feature in selected_features:
            importance_scores[feature] = 1.0
    
    # 如果指定了阈值，并且使用了基于特征重要性的方法
    if threshold is not None and method in ['mutual_info', 'f_regression']:
        # 根据阈值筛选特征
        selected_features = [feature for feature, score in importance_scores.items() 
                           if score >= threshold]
        
        # 更新选择的特征矩阵
        X_selected = X_numeric[selected_features].values
    
    # 返回选择后的特征矩阵和特征名称
    # 修复：使用X_numeric的索引而不是X的索引
    X_selected_df = pd.DataFrame(X_selected, index=X_numeric.index, columns=selected_features)
    
    logger.info(f"特征选择完成，选择了 {len(selected_features)} 个特征")
    return X_selected_df, selected_features 