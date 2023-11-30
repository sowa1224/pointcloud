import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/extracted_data.csv"
data = pd.read_csv(csv_file_path)

# データの準備 (三次元座標の列を選択)
X = data[['X', 'Y', 'Z'] ]
print(X)

# 探索するクラスタ数の範囲を指定
min_clusters = 2
max_clusters =7
    
# 各クラスタ数に対するシルエットスコアを計算
silhouette_scores = []
for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
# シルエットスコアのプロット
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()
