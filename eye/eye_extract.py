import sys
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from my_functions import *
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pyclustering.cluster import xmeans
from pyclustering.cluster.xmeans import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import splitting_type
import warnings


# 以降、numpy.warnings.filterwarnings の代わりに warnings.filterwarnings を使用

# 新しいCSVファイルとして保存
csv_file_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_0105_filtered.csv'
#csv_file_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0101_filtered.csv'
data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ','H','S','V']



# 彩度に基づいてデータを抽出
max_saturation = 180
min_saturation = 100
saturantion_range=(min_saturation,max_saturation)
extracted_data = extract_data_by_saturation(data, saturantion_range)

# 元のデータから抽出データを削除
data_without_extracted = data.drop(extracted_data.index)
# 座標データを抽出
points_original = data_without_extracted[['X', 'Y', 'Z']].values
# 色データを抽出
colors_original = data_without_extracted[['R', 'G', 'B']].values
# 法線データを抽出
normals_original = data_without_extracted[['nX', 'nY', 'nZ']].values

#各点をプロットする
#plot_csv(extracted_data)
#各点をHSV形式で表示
#plot_hsv_values(extracted_data)



# 座標データを抽出
points_extracted = extracted_data[['X', 'Y', 'Z']].values
colors_extracted = np.array([[1, 0, 0] for _ in range(len(points_extracted))] ) # 条件を満たす点を赤色で設定
normals_extracted = extracted_data[['nX', 'nY', 'nZ']].values

# Open3DのPointCloudオブジェクトを作成
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(points_original)
pcd_original.colors = o3d.utility.Vector3dVector(colors_original/250)
pcd_original.normals = o3d.utility.Vector3dVector(normals_original)

pcd_extracted = o3d.geometry.PointCloud()
pcd_extracted.points = o3d.utility.Vector3dVector(points_extracted)
pcd_extracted.colors = o3d.utility.Vector3dVector(colors_extracted)
pcd_extracted.normals = o3d.utility.Vector3dVector(normals_extracted)

# 可視化
o3d.visualization.draw_geometries([pcd_extracted, pcd_original])





#クラスター手法A  X＋シル
# データの抽出（X、Y、Z座標を使用）
points = extracted_data[['X', 'Y', 'Z']].values

# 探索するクラスタ数の範囲を指定
min_clusters = 2
max_clusters = 8

best_num_clusters = 1  # デフォルト値
best_silhouette_score = -1  # 最も高いシルエットスコア

# シルエット分析を実行し、最適なクラスタ数を見つける
for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # n_initの値を明示的に設定

    kmeans.fit(points)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(points, labels)
    print(num_clusters,silhouette_avg)
    
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_num_clusters = num_clusters
print(f"最適なクラスタ数: {best_num_clusters}")
# 最適なクラスタ数で再度K-meansクラスタリングを実行
kmeans = KMeans(n_clusters=best_num_clusters)
kmeans.fit(points)

# クラスタの中心座標
cluster_centers = kmeans.cluster_centers_

# 中心座標からの距離の閾値 (球状領域の半径)
radius_threshold = 0.005  # 任意の値を設定

# 各クラスタの中心座標に対して
for center in cluster_centers:
    # 球状領域内の点を抽出
    cluster_points = extracted_data[np.linalg.norm(extracted_data[['X', 'Y', 'Z']].values - center, axis=1) < radius_threshold]
    
    # 抽出した点の法線ベクトルを取得
    cluster_normals = cluster_points[['nX', 'nY', 'nZ']].values
    
    # 法線ベクトルの平均を計算
    mean_normal = np.mean(cluster_normals, axis=0)
    
    # クラスタごとの平均法線ベクトルを使用できます (mean_normal)
    print(f"クラスタ中心座標 {center}: 平均法線ベクトル {mean_normal}")

#関数の内容は以下のコードに示す
plot_cluster_result(cluster_centers,pcd_original,pcd_extracted)
"""""
pcd_class = o3d.geometry.PointCloud()
pcd_class.points = o3d.utility.Vector3dVector(cluster_centers)
colors_class = np.array([[0, 0, 1] for _ in range(len(cluster_centers))] ) # 条件を満たす点を赤色で設定
pcd_class.colors = o3d.utility.Vector3dVector(colors_class)

# ウィンドウを作成してデータを表示
o3d.visualization.draw_geometries([pcd_original, pcd_extracted,pcd_class])
"""""



#クラスター手法B
#エルボー法を利用した場合
# エルボー法で最適なクラスター数を見つける
min_clusters = 2
max_clusters = 8

# 各クラスタ数に対するSSEを保存するリスト
sse = []

# クラスタ数ごとにK-meansクラスタリングを実行し、SSEを計算
for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # n_initの値を明示的に設定
    kmeans.fit(points)
    sse.append(kmeans.inertia_)  # クラスタ内誤差平方和 (SSE)
"""""
# SSEをプロットしてエルボーを探す
plt.figure()
plt.plot(range(min_clusters, max_clusters + 1), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()
print(sse)
"""""

sse_diffs = []  # SSEの差の絶対値を保存
for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # n_initの値を明示的に設定
    kmeans.fit(points)
    sse = kmeans.inertia_  # クラスタ内誤差平方和 (SSE)
    sse_diffs.append(abs(sse - sse_diffs[-1]) if sse_diffs else 0)


# SSEの差の差分を計算
sse_diff_changes = [sse_diffs[i] - sse_diffs[i - 1] for i in range(1, len(sse_diffs))]

# 最適なクラスター数aを見つける
optimal_cluster_index = sse_diff_changes.index(min(sse_diff_changes))
optimal_cluster_count = min_clusters + optimal_cluster_index+1

print(f"最適なクラスター数: {optimal_cluster_count}")

# エルボー法の結果をもとに最適なクラスター数を選択
best_num_clusters=optimal_cluster_count
#best_num_clusters = int(input("最適なクラスター数は？"))  # エルボー法の結果から選択
# データの抽出（X、Y、Z座標を使用）
points = extracted_data[['X', 'Y', 'Z']].values

# K-meansクラスタリングを最適なクラスター数で実行
kmeans = KMeans(n_clusters=best_num_clusters, n_init=10)
kmeans.fit(points)

# クラスタの中心座標
cluster_centers = kmeans.cluster_centers_

# 中心座標からの距離の閾値 (球状領域の半径)
radius_threshold = 0.005  # 任意の値を設定

# 各クラスタの中心座標に対して
for center in cluster_centers:
    # 球状領域内の点を抽出
    cluster_points = extracted_data[np.linalg.norm(extracted_data[['X', 'Y', 'Z']].values - center, axis=1) < radius_threshold]

    # 抽出した点の法線ベクトルを取得
    cluster_normals = cluster_points[['nX', 'nY', 'nZ']].values

    # 法線ベクトルの平均を計算
    mean_normal = np.mean(cluster_normals, axis=0)

    # クラスタごとの平均法線ベクトルを使用できます (mean_normal)
    print(f"クラスタ中心座標 {center}: 平均法線ベクトル {mean_normal}")

plot_cluster_result(cluster_centers,pcd_original,pcd_extracted)









#クラスター手法C X-means
# X-meansクラスタリングを設定
# データの抽出（X、Y、Z座標を使用）
import pyclustering
from pyclustering.cluster import xmeans
import numpy as np
import pylab
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

#%% create data 3d
data =  extracted_data[['X', 'Y', 'Z']].values
print(data)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(data[..., 0], data[..., 1], data[..., 2])
plt.show()

#%% clustering
init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(data, 2).initialize()
xm = pyclustering.cluster.xmeans.xmeans(data, init_center, ccore=False)
xm.process()
clusters = xm.get_clusters()
pyclustering.utils.draw_clusters(data, clusters, display_result=False)
pylab.show()
# クラスターの中心座標を取得
cluster_centers = xm.get_centers()
print("クラスターの中心座標:")
for center in cluster_centers:
    print(center)

# 中心座標からの距離の閾値 (球状領域の半径)
radius_threshold = 0.005  # 任意の値を設定

# 各クラスタの中心座標に対して
for center in cluster_centers:
    # 球状領域内の点を抽出
    cluster_points = extracted_data[np.linalg.norm(extracted_data[['X', 'Y', 'Z']].values - center, axis=1) < radius_threshold]
    
    # 抽出した点の法線ベクトルを取得
    cluster_normals = cluster_points[['nX', 'nY', 'nZ']].values
    
    # 法線ベクトルの平均を計算
    mean_normal = np.mean(cluster_normals, axis=0)
    
    # クラスタごとの平均法線ベクトルを使用できます (mean_normal)
    print(f"クラスタ中心座標 {center}: 平均法線ベクトル {mean_normal}")

#plot_cluster_result(cluster_centers,pcd_original,pcd_extracted)


"""
initial_centers = [[0, 0, 0]]  # 初期中心座標のリスト（初期値を指定する場合）
xmeans_instance = xmeans(points_extracted, initial_centers, kmax=10)

# クラスタリングを実行
xmeans_instance.process()

# クラスタの中心座標を取得
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()

# クラスタリング結果を可視化
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, points_extracted)
visualizer.show()

# クラスタの中心座標を表示
for idx, center in enumerate(centers):
    print(f"クラスタ {idx} の中心座標: {center}")

# 中心座標からの距離の閾値 (球状領域の半径)
radius_threshold = 0.005  # 任意の値を設定

# 各クラスタの中心座標に対して
for center in cluster_centers:
    # 球状領域内の点を抽出
    cluster_points = extracted_data[np.linalg.norm(extracted_data[['X', 'Y', 'Z']].values - center, axis=1) < radius_threshold]
    
    # 抽出した点の法線ベクトルを取得
    cluster_normals = cluster_points[['nX', 'nY', 'nZ']].values
    
    # 法線ベクトルの平均を計算
    mean_normal = np.mean(cluster_normals, axis=0)
    
    # クラスタごとの平均法線ベクトルを使用できます (mean_normal)
    print(f"クラスタ中心座標 {center}: 平均法線ベクトル {mean_normal}")

plot_cluster_result(cluster_centers,pcd_original,pcd_extracted)
"""