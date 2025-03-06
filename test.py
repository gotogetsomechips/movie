import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import defaultdict
import jieba
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
import tensorflow as tf
import tensorflow_recommenders as tfrs

# 1. 读取电影短评文件夹下的所有csv文件
def load_all_reviews(folder_path):
    """
    读取指定文件夹中的所有CSV文件并合并为一个DataFrame
    """
    all_reviews = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                # 假设CSV文件的编码为UTF-8，可能需要根据实际情况调整
                df = pd.read_csv(file_path, encoding='utf-8')
                # 标准化列名
                if '用户名' in df.columns and '评论时间' in df.columns and 'IP属地' in df.columns and \
                   '评论分数' in df.columns and '有用数量' in df.columns and '短评内容' in df.columns:
                    # 统一列名
                    df = df.rename(columns={
                        '用户名': 'username',
                        '评论时间': 'review_time',
                        'IP属地': 'location',
                        '评论分数': 'rating',
                        '有用数量': 'useful_count',
                        '短评内容': 'content'
                    })
                    # 从文件名中提取电影名称作为ID（去掉扩展名）
                    movie_name = file.split('.')[0]  # 获取不含扩展名的文件名
                    df['movie_id'] = movie_name  # 使用完整文件名作为电影ID
                    all_reviews.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    if all_reviews:
        return pd.concat(all_reviews, ignore_index=True)
    else:
        return pd.DataFrame()

# 2. 数据预处理
def preprocess_data(df):
    """
    数据清洗和预处理
    """
    # 创建明确的副本
    df = df.copy()
    
    print(f"预处理前数据行数: {len(df)}")
    print(f"预处理前数据列: {df.columns.tolist()}")
    
    # 处理缺失值
    df = df.dropna(subset=['username', 'movie_id'])
    
    # 处理评分数据 - 提取数值部分（去掉"星"字）
    def extract_rating(rating_str):
        if pd.isna(rating_str):
            return None
        if isinstance(rating_str, (int, float)):
            return rating_str
        
        # 使用正则表达式提取数字部分
        rating_match = re.search(r'(\d+)', str(rating_str))
        if rating_match:
            return int(rating_match.group(1))
        return None
    
    df.loc[:, 'rating'] = df['rating'].apply(extract_rating)
    print(f"转换rating为数值型后的非空值数量: {df['rating'].notna().sum()}")
    
    df = df.dropna(subset=['rating'])
    print(f"删除rating缺失值后的行数: {len(df)}")
    
    # 时间转换
    def parse_time(time_str):
        try:
            if isinstance(time_str, str):
                return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            return None
        except:
            return None
    
    df['review_time'] = df['review_time'].apply(parse_time)
    # 确保所有电影ID都是字符串
    df['movie_id'] = df['movie_id'].astype(str) 
    # 提取用户和电影的特征
    # 用户特征: 评论数量, 平均评分, 评论文本长度平均值
    user_features = df.groupby('username').agg({
        'movie_id': 'count',
        'rating': 'mean',
        'content': lambda x: np.mean([len(str(text)) for text in x if isinstance(text, str)])
    }).rename(columns={
        'movie_id': 'review_count',
        'rating': 'avg_rating',
        'content': 'avg_content_length'
    }).reset_index()
    
    # 电影特征: 评论数量, 平均评分, 平均有用数
    movie_features = df.groupby('movie_id').agg({
        'username': 'count',
        'rating': 'mean',
        'useful_count': 'mean'
    }).rename(columns={
        'username': 'review_count',
        'rating': 'avg_rating',
        'useful_count': 'avg_useful'
    }).reset_index()
    
    return df, user_features, movie_features

# 3. 特征工程
def feature_engineering(df, user_features, movie_features):
    """
    为LightFM和TensorFlow Recommenders准备特征
    """
    print(f"特征工程前数据行数: {len(df)}")
    
    # 为每个用户和电影分配唯一ID
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    user_features['user_id'] = user_encoder.fit_transform(user_features['username'])
    movie_features['movie_id_encoded'] = movie_encoder.fit_transform(movie_features['movie_id'])
    
    # 映射回原始数据
    user_id_map = dict(zip(user_features['username'], user_features['user_id']))
    movie_id_map = dict(zip(movie_features['movie_id'], movie_features['movie_id_encoded']))
    
    df['user_id'] = df['username'].map(user_id_map)
    df['movie_id_encoded'] = df['movie_id'].map(movie_id_map)
    
    # 删除没有映射成功的行
    df = df.dropna(subset=['user_id', 'movie_id_encoded'])
    df['user_id'] = df['user_id'].astype(int)
    df['movie_id_encoded'] = df['movie_id_encoded'].astype(int)
    
    print(f"映射ID后的数据行数: {len(df)}")
    
    # 创建交互矩阵
    interactions = sparse.coo_matrix((
        df['rating'].astype(float).values,
        (df['user_id'].values, df['movie_id_encoded'].values)
    ))
    
    # 划分训练集和测试集
    train_interactions, test_interactions = train_test_split(
        df, test_size=0.2, random_state=42
    )
    
    train_matrix = sparse.coo_matrix((
        train_interactions['rating'].astype(float).values,
        (train_interactions['user_id'].values, train_interactions['movie_id_encoded'].values)
    ))
    
    test_matrix = sparse.coo_matrix((
        test_interactions['rating'].astype(float).values,
        (test_interactions['user_id'].values, test_interactions['movie_id_encoded'].values)
    ))
    
    # 修复用户特征：确保没有NaN值
    for col in ['review_count', 'avg_rating', 'avg_content_length']:
        # 填充NaN值为0
        user_features[col] = user_features[col].fillna(0)
        # 确保所有值都是有限的
        if not np.all(np.isfinite(user_features[col])):
            print(f"警告：列 {col} 包含无穷值，将替换为0")
            user_features[col] = np.where(np.isfinite(user_features[col]), user_features[col], 0)
    
    # 修复电影特征：确保没有NaN值
    for col in ['review_count', 'avg_rating', 'avg_useful']:
        # 填充NaN值为0
        movie_features[col] = movie_features[col].fillna(0)
        # 确保所有值都是有限的
        if not np.all(np.isfinite(movie_features[col])):
            print(f"警告：列 {col} 包含无穷值，将替换为0")
            movie_features[col] = np.where(np.isfinite(movie_features[col]), movie_features[col], 0)
    
    # 归一化连续特征
    for col in ['review_count', 'avg_rating', 'avg_content_length']:
        min_val = user_features[col].min()
        max_val = user_features[col].max()
        if max_val > min_val:  # 避免除以零
            user_features[col] = (user_features[col] - min_val) / (max_val - min_val)
        else:
            user_features[col] = 0  # 如果所有值相同，设为0
    
    for col in ['review_count', 'avg_rating', 'avg_useful']:
        min_val = movie_features[col].min()
        max_val = movie_features[col].max()
        if max_val > min_val:
            movie_features[col] = (movie_features[col] - min_val) / (max_val - min_val)
        else:
            movie_features[col] = 0
    
    # 检查用户特征中是否存在NaN
    print(f"用户特征中的NaN值: {user_features[['review_count', 'avg_rating', 'avg_content_length']].isna().sum()}")
    
    # 检查电影特征中是否存在NaN
    print(f"电影特征中的NaN值: {movie_features[['review_count', 'avg_rating', 'avg_useful']].isna().sum()}")
    
    # 构建用户特征矩阵
    user_features_matrix = sparse.csr_matrix(user_features[['review_count', 'avg_rating', 'avg_content_length']].values)
    
    # 构建电影特征矩阵
    movie_features_matrix = sparse.csr_matrix(movie_features[['review_count', 'avg_rating', 'avg_useful']].values)
    
    # 为TensorFlow Recommenders准备数据
    # 创建TensorFlow数据集
    tf_train_ratings = tf.data.Dataset.from_tensor_slices({
        "user_id": train_interactions['user_id'].values,
        "movie_id": train_interactions['movie_id_encoded'].values,
        "rating": train_interactions['rating'].values,
    })
    
    tf_test_ratings = tf.data.Dataset.from_tensor_slices({
        "user_id": test_interactions['user_id'].values,
        "movie_id": test_interactions['movie_id_encoded'].values,
        "rating": test_interactions['rating'].values,
    })
    
    # 获取唯一用户和电影ID
    unique_user_ids = np.unique(df['user_id'].values)
    unique_movie_ids = np.unique(df['movie_id_encoded'].values)
    
    return {
        'user_features': user_features,
        'movie_features': movie_features,
        'interactions': interactions,
        'train_matrix': train_matrix,
        'test_matrix': test_matrix,
        'user_features_matrix': user_features_matrix,
        'movie_features_matrix': movie_features_matrix,
        'tf_train_ratings': tf_train_ratings,
        'tf_test_ratings': tf_test_ratings,
        'unique_user_ids': unique_user_ids,
        'unique_movie_ids': unique_movie_ids,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'train_interactions': train_interactions,
        'test_interactions': test_interactions
    }

# 4.1 LightFM 模型
def build_lightfm_model(features_data):
    """
    构建和训练LightFM模型
    """
    # 检查输入数据
    print(f"训练矩阵维度: {features_data['train_matrix'].shape}")
    print(f"测试矩阵维度: {features_data['test_matrix'].shape}")
    print(f"用户特征矩阵维度: {features_data['user_features_matrix'].shape}")
    print(f"电影特征矩阵维度: {features_data['movie_features_matrix'].shape}")
    
    # 使用更少的组件和更简单的损失函数
    print("创建LightFM模型...")
    model = LightFM(no_components=10, loss='warp')
    
    # 训练模型，减少epoch数量，只用单线程
    print("开始训练模型...")
    model.fit(
        features_data['train_matrix'],
        user_features=features_data['user_features_matrix'],
        item_features=features_data['movie_features_matrix'],
        epochs=5,  # 减少到5个epoch
        num_threads=1,  # 只用1个线程
        verbose=True
    )
    
    print("模型训练完成!")
    
    # 评估模型
    print("评估模型...")
    train_precision = precision_at_k(
        model, 
        features_data['train_matrix'],
        user_features=features_data['user_features_matrix'],
        item_features=features_data['movie_features_matrix'],
        k=5
    ).mean()
    
    test_precision = precision_at_k(
        model, 
        features_data['test_matrix'],
        user_features=features_data['user_features_matrix'],
        item_features=features_data['movie_features_matrix'],
        k=5
    ).mean()
    
    train_auc = auc_score(
        model, 
        features_data['train_matrix'],
        user_features=features_data['user_features_matrix'],
        item_features=features_data['movie_features_matrix']
    ).mean()
    
    test_auc = auc_score(
        model, 
        features_data['test_matrix'],
        user_features=features_data['user_features_matrix'],
        item_features=features_data['movie_features_matrix']
    ).mean()
    
    print(f"LightFM - 训练集上的 Precision@5: {train_precision:.4f}")
    print(f"LightFM - 测试集上的 Precision@5: {test_precision:.4f}")
    print(f"LightFM - 训练集上的 AUC: {train_auc:.4f}")
    print(f"LightFM - 测试集上的 AUC: {test_auc:.4f}")
    
    # 预测指定用户的电影评分
    def predict_ratings(user_id, top_n=5):
        # 获取所有电影ID
        all_movie_ids = np.arange(len(features_data['unique_movie_ids']))
        
        # 需要将user_id转换为与all_movie_ids长度相同的数组
        user_ids = np.full(len(all_movie_ids), user_id, dtype=np.int32)
        
        # 为指定用户预测所有电影的评分
        predictions = model.predict(
            user_ids,
            all_movie_ids,
            user_features=features_data['user_features_matrix'],
            item_features=features_data['movie_features_matrix']
        )
        
        # 关键修改：将任意范围的分数映射到评分范围(例如1-5)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        if max_pred > min_pred:  # 避免除以零
            normalized_predictions = (predictions - min_pred) / (max_pred - min_pred) * 4 + 1
        else:
            normalized_predictions = np.ones_like(predictions) * 3  # 默认中等评分
        
        # 获取评分最高的N部电影
        top_items = np.argsort(-predictions)[:top_n]  # 仍使用原始预测进行排序
        
        # 转回原始电影ID
        original_movie_ids = [features_data['movie_features'].loc[features_data['movie_features']['movie_id_encoded'] == item_id, 'movie_id'].values[0] for item_id in top_items]
        clean_movie_names = [movie_id.replace(" 短评", "") for movie_id in original_movie_ids]

        # 然后返回清理后的名称
        return list(zip(clean_movie_names, normalized_predictions[top_items]))
    
    results = {
        'model': model,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'predict_function': predict_ratings
    }
    
    return results

# 4.2 TensorFlow Recommenders模型
class MovieRatingModel(tfrs.Model):
    def __init__(self, unique_user_ids, unique_movie_ids):
        super().__init__()
        
        # 4.2.1 模型架构设计
        # 定义用户和电影的嵌入层
        embedding_dimension = 32
        
        # 用户模型
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids,
                mask_token=None,
                num_oov_indices=1,
                output_mode='int'),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        
        # 电影模型
        self.movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_ids,
                mask_token=None,
                num_oov_indices=1,
                output_mode='int'),
            tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
        ])
        
        # 评分模型
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
        # 任务
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
    
    def call(self, features):
        # 获取用户和电影嵌入 - 根据输入数据类型做适当处理
        user_ids = features["user_id"]
        movie_ids = features["movie_id"]
        
        # 如果输入不是字符串类型，则转换
        if user_ids.dtype != tf.string:
            user_ids = tf.strings.as_string(user_ids)
        if movie_ids.dtype != tf.string:
            movie_ids = tf.strings.as_string(movie_ids)
        
        user_embedding = self.user_model(user_ids)
        movie_embedding = self.movie_model(movie_ids)
        
        # 连接嵌入
        x = tf.concat([user_embedding, movie_embedding], axis=1)
        
        # 预测评分
        return self.rating_model(x)
    
    def compute_loss(self, features, training=False):
        ratings = features.pop("rating")
        rating_predictions = self(features)
        return self.task(
            labels=ratings,
            predictions=rating_predictions,
        )

def build_tensorflow_recommenders_model(features_data):
    """
    构建和训练TensorFlow Recommenders模型
    """
    # 4.2.2 训练与优化策略
    # 转换数据类型
    unique_user_ids = tf.strings.as_string(features_data['unique_user_ids'])
    unique_movie_ids = tf.strings.as_string(features_data['unique_movie_ids'])
    
    # 准备批处理数据
    train_dataset = features_data['tf_train_ratings'].batch(8192).cache()
    test_dataset = features_data['tf_test_ratings'].batch(4096).cache()
    
    # 创建模型
    model = MovieRatingModel(unique_user_ids, unique_movie_ids)
    
    # 配置优化器
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    
    # 训练模型
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=5
    )
    
    # 评估模型
    evaluation = model.evaluate(test_dataset, return_dict=True)
    print(f"TensorFlow Recommenders - 测试集上的 RMSE: {evaluation['root_mean_squared_error']:.4f}")
    
    # 创建预测函数
    def predict_ratings_tf(user_id, top_n=5):
        # 创建测试数据
        user_ids = np.full(len(features_data['unique_movie_ids']), user_id)
        movie_ids = features_data['unique_movie_ids']
        
        # 进行预测
        test_data = tf.data.Dataset.from_tensor_slices({
            "user_id": user_ids,
            "movie_id": movie_ids,
        }).batch(128)
        
        predictions = model.predict(test_data)
        
        # 获取评分最高的N部电影
        movie_indices = np.argsort(-predictions.flatten())[:top_n]
        top_movie_ids = [features_data['unique_movie_ids'][idx] for idx in movie_indices]
        
        # 转回原始电影ID
        original_movie_ids = [features_data['movie_features'].loc[features_data['movie_features']['movie_id_encoded'] == movie_id, 'movie_id'].values[0] for movie_id in top_movie_ids]
        
        # 提取电影名称并去除"短评"后缀
        movie_names = [movie_id.replace(" 短评", "") for movie_id in original_movie_ids]
        
        return list(zip(movie_names, predictions.flatten()[movie_indices]))
    
    results = {
        'model': model,
        'history': history,
        'evaluation': evaluation,
        'predict_function': predict_ratings_tf
    }
    
    return results

# 4.3 预测结果与对比分析
def compare_models(lightfm_results, tf_results, features_data):
    """
    对比两个模型的性能和预测结果
    """
    # 随机选取一些用户进行预测比较
    random_users = np.random.choice(features_data['unique_user_ids'], 5)
    
    # 4.3.2 实验结果
    print("\n=== 模型性能对比 ===")
    print(f"LightFM - 测试集上的 Precision@5: {lightfm_results['test_precision']:.4f}")
    print(f"LightFM - 测试集上的 AUC: {lightfm_results['test_auc']:.4f}")
    print(f"TensorFlow Recommenders - 测试集上的 RMSE: {tf_results['evaluation']['root_mean_squared_error']:.4f}")
    
    print("\n=== 预测结果对比 ===")
    for user_id in random_users:
        print(f"\n用户ID: {user_id}")
        print("LightFM推荐:")
        lightfm_preds = lightfm_results['predict_function'](user_id)
        for movie_name, score in lightfm_preds:
            print(f"  电影: {movie_name}, 预测评分: {score:.2f}")
        
        print("TensorFlow Recommenders推荐:")
        tf_preds = tf_results['predict_function'](user_id)
        for movie_name, score in tf_preds:
            print(f"  电影: {movie_name}, 预测评分: {score:.2f}")
    
    # 4.3.3 结果分析
    # 计算两个模型的推荐重叠率
    overlap_counts = []
    for user_id in random_users:
        lightfm_movies = set([m for m, _ in lightfm_results['predict_function'](user_id)])
        tf_movies = set([m for m, _ in tf_results['predict_function'](user_id)])
        overlap = len(lightfm_movies.intersection(tf_movies))
        overlap_counts.append(overlap / 5.0)  # 5是top_n
    
    avg_overlap = np.mean(overlap_counts)
    print(f"\n两个模型的平均推荐重叠率: {avg_overlap:.2f}")
    
    # 绘制模型训练历史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tf_results['history'].history['root_mean_squared_error'])
    plt.plot(tf_results['history'].history['val_root_mean_squared_error'])
    plt.title('TensorFlow Recommenders 模型 RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='upper right')
    
    # 模型偏好分析
    plt.subplot(1, 2, 2)
    # 分析LightFM和TF模型的评分分布差异
    lightfm_ratings = []
    tf_ratings = []
    
    for user_id in features_data['unique_user_ids'][:100]:  # 使用前100个用户以避免计算过多
        lightfm_preds = lightfm_results['predict_function'](user_id)
        tf_preds = tf_results['predict_function'](user_id)
        
        for _, score in lightfm_preds:
            lightfm_ratings.append(score)
        
        for _, score in tf_preds:
            tf_ratings.append(score)
    
    plt.hist(lightfm_ratings, alpha=0.5, label='LightFM')
    plt.hist(tf_ratings, alpha=0.5, label='TensorFlow')
    plt.title('预测评分分布')
    plt.xlabel('评分')
    plt.ylabel('频次')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 分析结论
    analysis = """
    """
    
    print(analysis)
    
    return {
        'overlap_rate': avg_overlap,
        'lightfm_ratings': lightfm_ratings,
        'tf_ratings': tf_ratings,
        'analysis': analysis
    }

# 主函数
def main(folder_path):
    print("1. 读取电影短评数据...")
    df = load_all_reviews(folder_path)
    
    if df.empty:
        print("没有找到有效的CSV文件或数据为空，请检查文件路径和格式")
        return
    
    print(f"读取了 {len(df)} 条评论数据，涉及 {df['movie_id'].nunique()} 部电影")
    
    print("\n2. 数据预处理...")
    df, user_features, movie_features = preprocess_data(df)
    
    # 添加检查
    if df.empty or user_features.empty or movie_features.empty:
        print("预处理后数据为空，请检查数据格式")
        return
    
    print(f"预处理后数据: {len(df)} 条评论, {len(user_features)} 个用户, {len(movie_features)} 部电影")
    
    print("\n3. 特征工程...")
    features_data = feature_engineering(df, user_features, movie_features)
    
    print("\n4.1 构建和训练LightFM模型...")
    lightfm_results = build_lightfm_model(features_data)
    
    print("\n4.2 构建和训练TensorFlow Recommenders模型...")
    tf_results = build_tensorflow_recommenders_model(features_data)
    
    print("\n4.3 预测结果与对比分析...")
    comparison_results = compare_models(lightfm_results, tf_results, features_data)
    
    print("\n完成所有任务!")
    return df, features_data, lightfm_results, tf_results, comparison_results

if __name__ == "__main__":
    # 使用示例
    folder_path = "./电影短评"  # 替换为实际的文件夹路径
    main(folder_path)