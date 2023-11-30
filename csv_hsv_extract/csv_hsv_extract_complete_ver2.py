import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import sys
from my_functions import *
import colorsys
import os

# my_functions.py のパスを取得します。このパスは my_functions.py があるディレクトリの絶対パスである必要があります。
my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program"
# sys.path に my_functions.py のディレクトリのパスを追加します。
sys.path.append(my_functions_path)

# CSVファイルが格納されているディレクトリ
csv_directory = "/home/sowa/prog/pointcloud/csv/potato_1118/peel_copy/"

# CSV_filteredファイルを保存するディレクトリ
csv_output_directory = "/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered_copy/"

# 半径外れ値除去のパラメータ
radius, radius_1 = 0.07, 0.008
min_points, min_points_1 = 80, 80

# 抽出する色相の範囲
min_hue, max_hue = 12, 50
hue_range = (min_hue, max_hue)

# 抽出する彩度の範囲
min_saturation, max_saturation = 40, 255
saturation_range = (min_saturation, max_saturation)

# 抽出する明度の範囲
min_value, max_value = 20, 80
value_range = (min_value, max_value)

# ループで各ファイルを処理
for file_name in os.listdir(csv_directory):
    if file_name.endswith(".csv"):
        # CSVファイルのパス
        csv_file_path = os.path.join(csv_directory, file_name)

        # CSV_filteredファイルのパス
        csv_filtered_file_path = os.path.join(csv_output_directory, file_name.replace(".csv", "_filtered.csv"))
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ','H','S','V']
        hue_extract_data = extract_data_by_hue(data, hue_range)

        # 色相に基づいてデータを抽出
        hue_extract_data = extract_data_by_hue(data, hue_range)
        # 彩度に基づいてデータを抽出
        extracted_hue_satura = extract_data_by_saturation(hue_extract_data, saturation_range)
        # 明度に基づいてデータを抽出
        extracted_hue_satura_value = extract_data_by_value(extracted_hue_satura, value_range)

        # データを抽出
        points = extracted_hue_satura_value[['X', 'Y', 'Z']].values
        colors = extracted_hue_satura_value[['R', 'G', 'B']].values # 0-255の範囲から0-1の範囲に変換
        normals = extracted_hue_satura_value[['nX', 'nY', 'nZ']].values
        hsv = extracted_hue_satura_value[['H', 'S', 'V']].values

        # Open3DのPointCloudオブジェクトを作成
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # 半径外れ値除去を実行
        filtered_pcd = remove_radius_outliers(pcd, radius, min_points)
        filtered_pcd_1 = remove_radius_outliers(filtered_pcd, radius_1, min_points_1)

        # フィルタリング後のデータを取得
        filtered_points = np.asarray(filtered_pcd_1.points)
        filtered_colors = np.asarray(filtered_pcd_1.colors) # 0-1の範囲から0-255の範囲に変換
        filtered_normals = np.asarray(filtered_pcd_1.normals)

        # RGBカラーをHSVに変換（0から1の範囲,HSVを0から255の範囲に変換
        hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in filtered_colors])
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
        # CSVファイルの保存
        df.to_csv(csv_filtered_file_path, index=False, header=False)
        print(f"点群データがCSVファイルに保存されました。ファイル名: {csv_filtered_file_path}, サイズ: {df.shape}")

print("全ての点群データの変換が完了しました。")











