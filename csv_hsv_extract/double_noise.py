import numpy as np
import pandas as pd
import cv2
import open3d as o3d

csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_filtered_1025_1.csv"
#csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_filtered_1017_peeling1.csv"
df = pd.read_csv(csv_file_path, header=None)
df.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']

# RGBデータをHSVデータに変換する関数
def rgb_to_hsv(row):
    rgb = np.array(row[['R', 'G', 'B']])
    rgb = rgb * 255
    hsv = cv2.cvtColor(rgb.astype('uint8').reshape(1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv[0, 0]


# 彩度に基づいてデータを抽出する関数
def extract_data_by_saturation(data, saturation_range):
    min_saturation, max_saturation = saturation_range
    hsv_data = data[['R', 'G', 'B']].apply(rgb_to_hsv, axis=1)
    saturation_values = hsv_data.apply(lambda x: x[1])  # 彩度はHSVの2番目の要素
    indices = (saturation_values >= min_saturation) & (saturation_values <= max_saturation)
    extracted_data = data[indices]
    return extracted_data

# 抽出する彩度の範囲を指定
min_saturation = 70  # 最小彩度
max_saturation = 255  # 最大彩度

saturation_range = (min_saturation, max_saturation)

# 彩度に基づいてデータを抽出
extracted_data = extract_data_by_saturation(df, saturation_range)
extracted_data.to_csv("/home/sowa/prog/pointcloud/csv/potato_hsv_extarct_1025_peeling1_1.csv", index=False)

csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_extarct_1025_peeling1_1.csv"

df = pd.read_csv(csv_file_path)

# 座標データを抽出
points = df[['X', 'Y', 'Z']].values

# 色データを抽出
colors = df[['R', 'G', 'B']].values # 0-255の範囲から0-1の範囲に変換

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可視化
o3d.visualization.draw_geometries([pcd])

csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_extarct_1025_peeling1_1.csv"
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# 座標データを抽出
points = data[:, :3]

# RGBデータを抽出
colors = data[:, 3:6] # 0-255の範囲から0-1の範囲に変換

# 法線データを抽出
normals = data[:, 6:]

# 半径外れ値除去法のパラメータを設定
radius = 0.0078 # 半径の閾値
min_points = 55  # 最小点数の閾値

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)
#もしpcd.clorsがなければ、可視化される点群データは色は含まれる

# 半径外れ値除去法を適用
pcd_filtered, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

# 可視化
o3d.visualization.draw_geometries([pcd_filtered])

# フィルタリング後の座標データを取得
filtered_points = np.asarray(pcd_filtered.points)

# フィルタリング後のRGBデータを取得
filtered_colors = np.asarray(pcd_filtered.colors)

# フィルタリング後の法線データを取得
filtered_normals = np.asarray(pcd_filtered.normals)

# フィルタリング後のデータを結合
filtered_data = np.concatenate((filtered_points, filtered_colors, filtered_normals), axis=1)

# 新しいCSVファイルとして保存
save_path = "/home/sowa/prog/pointcloud/csv/potato_filtered_1025_peeling1_doublenoise_1.csv"
np.savetxt(save_path, filtered_data, delimiter=',', header='X,Y,Z,R,G,B,nX,nY,nZ', comments='')