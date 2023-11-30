import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

output_csv_path = "/home/sowa/prog/pointcloud/csv/extracted_data1025_1.csv"
df = pd.read_csv(output_csv_path)
Numpy_df = df.values

# サンプルの点群データ（各点の(x, y, z)座標）
point_cloud = Numpy_df[:, :3]

# 探索距離の閾値
distance_threshold = 0.018

# ポイント間の距離行列を計算
distance_matrix = distance.cdist(point_cloud, point_cloud, 'euclidean')

# クラスタリングのためのデータ構造を初期化
visited = [False] * len(point_cloud)
clusters = []

# 距離行列を元にクラスタリング（クラスタ数が不明）
for i in range(len(point_cloud)):
    if visited[i]:
        continue
    cluster = [i]
    visited[i] = True

    for j, dist in enumerate(distance_matrix[i]):
        if not visited[j] and dist <= distance_threshold:
            cluster.append(j)
            visited[j] = True

    clusters.append(cluster)

# クラスタリング結果を表示
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, cluster in enumerate(clusters):
    cluster_points = point_cloud[cluster]
    x, y, z = cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]
    ax.scatter(x, y, z, label=f"Cluster {i + 1}")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()
plt.show()
