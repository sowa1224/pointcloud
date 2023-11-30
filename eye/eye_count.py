import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program/"
# sys.path に my_functions.py のディレクトリのパスを追加
sys.path.append(my_functions_path)
from my_functions import *


# 抽出したデータを新しいCSVファイルに保存
output_csv_file = "/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_0201_filtered.csv"
data = pd.read_csv(output_csv_file,header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

# 抽出する彩度の範囲
min_saturation, max_saturation = 100,140
saturation_range = (min_saturation,max_saturation)

# 抽出する明度の範囲
min_value, max_value = 90,160
value_range = (min_value,max_value)


# オリジナル画像をプロット
def plot_original(data_path,index):
    # データの読み込みと抽出
    output_csv_file = f"/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_020{index}_filtered.csv"
    data = pd.read_csv(output_csv_file, header=None)
    data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
    # プロット
    plt.subplot(3, 3, index)  # サブプロットの位置を設定
    plt.scatter(data[['X']].values, data[['Y']], c=data[['R', 'G', 'B']].values / 255, s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
# データを順番にプロット
plt.figure(figsize=(12, 12))  # 新しい図を作成
for index in range(1, 10):
    plot_original("/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_020",index)
plt.tight_layout()
plt.show()


# クラスタリングおよびプロットを行う関数
def plot_clusters(data_path, min_saturation, max_saturation, index):
    # データの読み込みと抽出
    output_csv_file = f"/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_020{index}_filtered.csv"
    data = pd.read_csv(output_csv_file, header=None)
    data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
    extracted_data_s = extract_data_by_saturation(data, saturation_range)
    extracted_data_v = extract_data_by_value(data, value_range)
    # extracted_data_sとextracted_data_vを結合
    combined_data = pd.concat([extracted_data_s, extracted_data_v])

    # 重複する行を除外
    extracted_data = combined_data.drop_duplicates()


    coordinates = extracted_data[['X', 'Y']].values
    
    # クラスタリング
    max_clusters = 7
    best_silhouette_score = -1
    optimal_num_clusters = 1

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
        silhouette_avg = silhouette_score(coordinates, cluster_labels)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            optimal_num_clusters = n_clusters

    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans.fit(coordinates)
    cluster_centers = kmeans.cluster_centers_
    
    # プロット
    plt.subplot(3, 3, index)  # サブプロットの位置を設定
    plt.scatter(data[['X']].values, data[['Y']], c=data[['R', 'G', 'B']].values / 255, s=5)
    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c="black", marker='o', label=f"Cluster {i+1} Center", s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Cluster Plot {index}')

# データを順番にプロット
plt.figure(figsize=(12, 12))  # 新しい図を作成
for index in range(1, 10):
    plot_clusters("/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_020", 100, 140, index)

plt.tight_layout()
plt.show()




# X-meansクラスタリングを関数として定義
def perform_xmeans_clustering(x):
    initial_centers = kmeans_plusplus_initializer(x, 2).initialize()
    xmeans_instance = xmeans(x, initial_centers)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return centers

def plot_cluster_results(data, centers, ax):
    ax.scatter(data[['X']].values, data[['Y']], c=data[['R', 'G', 'B']].values/255, s=5)
    for i, center in enumerate(centers):
        ax.scatter(center[0], center[1], c="black", marker='o', label=f"Cluster {i+1} Center", s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# 3x3のサブプロットを作成して各ファイルのクラスタリング結果をプロット
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for index in range(1, 10):
    output_csv_file = f"/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_020{index}_filtered.csv"
    data = pd.read_csv(output_csv_file, header=None)
    data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
    extracted_data_s = extract_data_by_saturation(data, saturation_range)
    extracted_data_v = extract_data_by_value(data, value_range)
    # extracted_data_sとextracted_data_vを結合
    combined_data = pd.concat([extracted_data_s, extracted_data_v])

    # 重複する行を除外
    extracted_data = combined_data.drop_duplicates()
    coordinates = extracted_data[['X', 'Y']].values

    centers = perform_xmeans_clustering(coordinates)
    #print(f"クラスター数は{len(centers)}")
    #print("Cluster centers:")
    #for center in centers:
    #    print(center)

    # 3x3のサブプロットのそれぞれにプロット
    row = (index - 1) // 3
    col = (index - 1) % 3
    plot_cluster_results(data, centers, axs[row, col])

plt.tight_layout()
plt.show()


""""
# シルエット分析による最適なクラスター数の選択
max_clusters = 7
best_silhouette_score = -1
optimal_num_clusters = 1  # 最小のクラスター数を初期化

for n_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coordinates)
    silhouette_avg = silhouette_score(coordinates, cluster_labels)
    
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        optimal_num_clusters = n_clusters

# K-meansでクラスタリングを実行し、各クラスターの中心座標を取得
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans.fit(coordinates)
cluster_centers = kmeans.cluster_centers_
print(f"クラスター数は{len(cluster_centers)}")
print("Cluster centers:")
for center in cluster_centers:
    print(center)

plt.scatter(data[['X']].values,data[['Y']],c=data[['R','G','B']].values/255, s=5)
# 各クラスターの中心座標をプロット
for i, center in enumerate(cluster_centers):
    plt.scatter(center[0], center[1], c="black", marker='o', label=f"Cluster {i+1} Center", s=100)

plt.scatter(data[['X']].values,data[['Y']],c=data[['R','G','B']].values/255, s=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""


""""



# 探索するクラスタ数の範囲を指定
min_clusters = 3
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


pcd_class = o3d.geometry.PointCloud()
pcd_class.points = o3d.utility.Vector3dVector(cluster_centers)
colors_class = np.array([[0, 0, 1] for _ in range(len(cluster_centers))] ) # 条件を満たす点を赤色で設定
pcd_class.colors = o3d.utility.Vector3dVector(colors_class)

# ウィンドウを作成してデータを表示
o3d.visualization.draw_geometries([pcd_original, pcd_extracted,pcd_class])

"""