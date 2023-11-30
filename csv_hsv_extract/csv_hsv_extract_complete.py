import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import sys
from my_functions import *
import colorsys


# my_functions.py のパスを取得します。このパスは my_functions.py があるディレクトリの絶対パスである必要があります。
my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program"
# sys.path に my_functions.py のディレクトリのパスを追加します。
sys.path.append(my_functions_path)
#テスト用　成功すればHellow Alice!が出力される
result = greet("Alice")
print(result)


#ここからが本来の内容
#csv_file_path = '/home/sowa/prog/pointcloud/csv/potato_1113/peel/potato_peel_1113_0.csv'
#csv_file_path= '/home/sowa/prog/pointcloud/csv/potato_1113/unpeel/potato_unpeel_1113_3.csv'
csv_file_path ="/home/sowa/prog/pointcloud/csv/potato_1114potato_unpeel_3.csv"
data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ','H','S','V']

# 抽出する色相の範囲を指定する
min_hue = 12
max_hue = 50
hue_range = (min_hue, max_hue)


# 色相に基づいてデータを抽出
hue_extract_data = extract_data_by_hue(data, hue_range)

# 座標データを抽出
points = hue_extract_data[['X', 'Y', 'Z']].values

# 色データを抽出
colors = hue_extract_data[['R', 'G', 'B']].values # 0-255の範囲から0-1の範囲に変換

# 法線データを抽出
normals = hue_extract_data[['nX', 'nY', 'nZ']].values

#HSVデータを抽出
hsv = hue_extract_data[['H', 'S', 'V']].values

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 可視化
#o3d.visualization.draw_geometries([pcd])


# 抽出する彩度の範囲を指定
#皮むき前なら110~255 皮むき後なら
min_saturation = 40  # 最小彩度
max_saturation = 255  # 最大彩度
saturation_range = (min_saturation, max_saturation)

# 彩度に基づいてデータを抽出
extracted_hue_satura = extract_data_by_saturation(hue_extract_data, saturation_range)
#print(extracted_hue_satura)

# 座標データを抽出
points = extracted_hue_satura[['X', 'Y', 'Z']].values

# 色データを抽出
colors = extracted_hue_satura[['R', 'G', 'B']].values # 0-255の範囲から0-1の範囲に変換

# 法線データを抽出
normals = extracted_hue_satura[['nX', 'nY', 'nZ']].values

#HSVデータを抽出
hsv = extracted_hue_satura[['H', 'S', 'V']].values

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 可視化
#o3d.visualization.draw_geometries([pcd])

#plot_hsv_values(extracted_hue_satura)


# 抽出するvalueの範囲を指定
#皮むき前なら110~255 皮むき後なら
min_value= 20  # 最小彩度
max_value = 80  # 最大彩度
value_range = (min_value, max_value)

# 彩度に基づいてデータを抽出
extracted_hue_satura_value = extract_data_by_value(extracted_hue_satura, saturation_range)
#print(extracted_hue_satura)

# 座標データを抽出
points = extracted_hue_satura_value[['X', 'Y', 'Z']].values

# 色データを抽出
colors = extracted_hue_satura_value[['R', 'G', 'B']].values # 0-255の範囲から0-1の範囲に変換

# 法線データを抽出
normals = extracted_hue_satura_value[['nX', 'nY', 'nZ']].values

#HSVデータを抽出
hsv = extracted_hue_satura_value[['H', 'S', 'V']].values

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 可視化
#o3d.visualization.draw_geometries([pcd])

#plot_hsv_values(extracted_hue_satura_value)


# 半径外れ値除去のパラメータを設定
radius,radius_1 = 0.07,0.008  # 半径内に何点以上の点があるか
min_points,min_points_1 = 80,90  # 半径内に何点以上の点がある場合に除去しないか

# 半径外れ値除去を実行
filtered_pcd = remove_radius_outliers(pcd, radius, min_points)
filtered_pcd_1 = remove_radius_outliers(filtered_pcd, radius_1, min_points_1)

# フィルタリング後の座標データを取得
filtered_points = np.asarray(filtered_pcd_1.points)

# フィルタリング後のRGBデータを取得
filtered_colors = np.asarray(filtered_pcd_1.colors) # 0-1の範囲から0-255の範囲に変換

# フィルタリング後の法線データを取得
filtered_normals = np.asarray(filtered_pcd_1.normals)



# RGBカラーをHSVに変換（0から1の範囲）
hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in filtered_colors])

# HSVを0から255の範囲に変換
filtered_hsv = (hsv_colors * 255).astype(int)

# フィルタリング後のデータを結合
filtered_data = np.concatenate((filtered_points, filtered_colors, filtered_normals, filtered_hsv), axis=1)
# 抽出した点群データをDataFrameに変換
df = pd.DataFrame({
    'X': filtered_points[:, 0],
    'Y': filtered_points[:, 1],
    'Z': filtered_points[:, 2],
    'R': filtered_colors[:, 0]*255,
    'G': filtered_colors[:, 1]*255,
    'B': filtered_colors[:, 2]*255,
    'nX': filtered_normals[:, 0],
    'nY': filtered_normals[:, 1],
    'nZ': filtered_normals[:, 2],
    'H': filtered_hsv[:, 0],  # 追加: 色相
    'S': filtered_hsv[:, 1],  # 追加: 彩度
    'V': filtered_hsv[:, 2]   # 追加: 明度
})




# CSVファイルとして保存（ヘッダー行を含めて保存）
#save_path='/home/sowa/prog/pointcloud/csv/potato_unpeel_3_filtered.csv'
#save_path='/home/sowa/prog/pointcloud/csv/potato_1113/peel/potato_peel_1113_0_filtered.csv'
save_path='/home/sowa/prog/pointcloud/csv/potato_1113/unpeel_filtered/potato_unpeel_1113_3_filtered.csv'
df.to_csv(save_path, index=False, header=False)  # headerをTrueに変更
print("点群データがCSVファイルに保存されました。")
print("CSVファイルの大きさは", df.shape)

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

# 可視化
#o3d.visualization.draw_geometries([pcd])

data = pd.read_csv(save_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ','H','S','V']


#plot_hsv_values(data)
