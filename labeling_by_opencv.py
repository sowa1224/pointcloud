import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score

# CSVファイルからデータを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1025.csv"
#csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1025_1.csv"
df = pd.read_csv(csv_file_path)

def rgb_to_hsv(row):
    rgb = np.array(row[['R', 'G', 'B']])
    rgb = rgb * 255
    hsv = cv2.cvtColor(rgb.astype('uint8').reshape(1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv[0, 0]

# HSVデータを計算し、新しい列としてデータフレームに追加
df['HSV'] = df.apply(rgb_to_hsv, axis=1)
# HSVデータを取得
hsv_data = np.array(df['HSV'].to_list())
h_values = hsv_data[:, 0]  # 色相 (Hue)
s_values = hsv_data[:, 1]  # 彩度 (Saturation)
v_values = hsv_data[:, 2]  # 明度 (Value)


# 3つのグラフを表示
#plt.figure(figsize=(12, 4))

# 点の番号を取得
point_numbers = np.arange(len(h_values))



# CSVファイルからデータを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1025.csv"
#csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1025_1.csv"
df = pd.read_csv(csv_file_path)

# RGBデータをHSVデータに変換する関数
def rgb_to_hsv(row):
    rgb = np.array(row[['R', 'G', 'B']])
    rgb = rgb * 255
    hsv = cv2.cvtColor(rgb.astype('uint8').reshape(1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv[0, 0]

# 彩度に基づいてデータを抽出
def extract_data_by_saturation(data, min_saturation):
    hsv_data = data[['R', 'G', 'B']].apply(rgb_to_hsv, axis=1)
    saturation_values = hsv_data.apply(lambda x: x[1])  # 彩度はHSVの2番目の要素
    indices = saturation_values >= min_saturation
    extracted_data = data[indices]
    return extracted_data

# 彩度に基づいてデータを抽出
min_saturation = 160
extracted_data = extract_data_by_saturation(df, min_saturation)

# 抽出したデータを新しいデータフレームに格納
new_df = extracted_data[['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']]


# 抽出したデータを新しいCSVファイルに保存
output_csv_file = "/home/sowa/prog/pointcloud/csv/extracted_data1025.csv"
#output_csv_file = "/home/sowa/prog/pointcloud/csv/extracted_data1025_1.csv"
extracted_data.to_csv(output_csv_file, index=False)

# 新しいデータフレームを表示
#print(new_df)
#print(np.size(new_df))

# 座標データを抽出
points_original = df[['X', 'Y', 'Z']].values
points_extracted = extracted_data[['X', 'Y', 'Z']].values

# 色データを設定
#colors_original = np.array([[0, 0, 1] for _ in range(len(points_original))] ) # 元の点を青色で設定
colors_original = df[['R', 'G', 'B']].values 
colors_extracted = np.array([[1, 0, 0] for _ in range(len(points_extracted))] ) # 条件を満たす点を赤色で設定

# Open3DのPointCloudオブジェクトを作成
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(points_original)
pcd_original.colors = o3d.utility.Vector3dVector(colors_original)

pcd_extracted = o3d.geometry.PointCloud()
pcd_extracted.points = o3d.utility.Vector3dVector(points_extracted)
pcd_extracted.colors = o3d.utility.Vector3dVector(colors_extracted)

# 可視化
o3d.visualization.draw_geometries([pcd_original, pcd_extracted])


extracted_points = extracted_data[['X', 'Y', 'Z']].values



# データの抽出（X、Y、Z座標を使用）
points = extracted_data[['X', 'Y', 'Z']].values

