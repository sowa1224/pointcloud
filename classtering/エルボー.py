import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# CSVファイルからデータを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/extracted_data1025.csv"
data = pd.read_csv(csv_file_path)

# データの準備 (1列目から3列目のデータを選択)
X = data[['X', 'Y', 'Z']]

# クラスタ数の候補
cluster_range = range(3, 8)
inertia = []  # クラスタ内の平均二乗誤差（SSE）を保存するリスト

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)  # SSEをリストに追加

# エルボ法の結果をプロット
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal Cluster Number')
plt.grid(True)
plt.show()
