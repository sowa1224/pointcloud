import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

def brightness(data):
    rgb=np.array(data[['R', 'G', 'B']])
    R,G,B = rgb[:,0],rgb[:,1],rgb[:,2]
    bright = 0.299 * R + 0.587 * G + 0.114 * B
    return bright


# RGBデータをHSVデータに変換する関数
def rgb_to_hsv(row):
    rgb = np.array(row[['R', 'G', 'B']])
    rgb = rgb * 255
    hsv = cv2.cvtColor(rgb.astype('uint8').reshape(1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv[0, 0]

# 色相に基づいてデータを抽出する関数
def extract_data_by_hue(data, hue_range):
    min_hue, max_hue = hue_range
    indices = (data['H'] >= min_hue) & (data['H'] <= max_hue)
    extracted_data = data[indices]
    return extracted_data


# 彩度に基づいてデータを抽出する関数
def extract_data_by_saturation(data, saturation_range):
    min_saturation, max_saturation = saturation_range
    indices = (data['S'] >= min_saturation) & (data['S'] <= max_saturation)
    extracted_data = data[indices]
    return extracted_data

# valueに基づいてデータを抽出する関数
def extract_data_by_value(data, value_range):
    min_value, max_value =value_range
    indices = (data['V'] >= min_value) & (data['V'] <= max_value)
    extracted_data = data[indices]
    return extracted_data


# 半径外れ値除去を行う関数
def remove_radius_outliers(pcd, radius, min_points):
    cl, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    return cl

# 半径外れ値除去を行う関数
def remove_radius_outliers_1(data, radius, min_points):

# ノイズを除去するための空のリストを作成
    filtered_data = []

# データ内の各点に対して処理を実行
    for i, point in data.iterrows():
        point_xyz = point[['X', 'Y', 'Z']].values  # X、Y、Z座標を抽出

    # 各点からの距離を計算
        distances = np.linalg.norm(data[['X', 'Y', 'Z']].values - point_xyz, axis=1)

    # 半径内の点の数を数える
        within_radius = len(np.where(distances <= radius)[0])

    # 半径内の点が最低の点の数以上の場合にノイズでないと判断
        if within_radius >= min_points:
            filtered_data.append(point)

# ノイズが除去された点群データを表示または保存
    filtered_data = pd.DataFrame(filtered_data)
    return filtered_data

def greet(name):
    return f"Hello, {name}!"

def plot_hsv_values(data):

    h_values = data['H']  # 色相 (Hue)
    s_values = data['S'] # 彩度 (Saturation)
    v_values = data['V'] # 明度 (Value)

# 各点のHSVデータを表示
    plt.figure(figsize=(12, 4))

# 点の番号を取得
    point_numbers = np.arange(len(h_values))

# 色相 (Hue) のプロット
    plt.subplot(131)
    plt.scatter(point_numbers, h_values, c=h_values, cmap='hsv', marker='.')
    plt.xlabel("Point Index")
    plt.ylabel("Hue (H)")
    plt.title("Hue (H) Values")

# 彩度 (Saturation) のプロット
    plt.subplot(132)
    plt.scatter(point_numbers, s_values, c=s_values, cmap='hsv', marker='.')
    plt.xlabel("Point Index")
    plt.ylabel("Saturation (S)")
    plt.title("Saturation (S) Values")

# 明度 (Value) のプロット
    plt.subplot(133)
    plt.scatter(point_numbers, v_values, c=v_values, cmap='hsv', marker='.')
    plt.xlabel("Point Index")
    plt.ylabel("Value (V)")
    plt.title("Value (V) Values")

    plt.tight_layout()
    plt.show()

def plot_csv(csv_data):

    # 座標データを抽出
    points = csv_data[['X', 'Y', 'Z']].values

    # 色データを抽出
    colors = csv_data[['R', 'G', 'B']].values/255 # 0-255の範囲から0-1の範囲に変換

    # 法線データを抽出
    normals = csv_data[['nX', 'nY', 'nZ']].values 

    # Open3DのPointCloudオブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 可視化
    o3d.visualization.draw_geometries([pcd])


def plot_cluster_result(cluster_centers,pcd_original,pcd_extracted):
    pcd_class = o3d.geometry.PointCloud()
    pcd_class.points = o3d.utility.Vector3dVector(cluster_centers)
    colors_class = np.array([[0, 0, 1] for _ in range(len(cluster_centers))] ) # 条件を満たす点を赤色で設定
    pcd_class.colors = o3d.utility.Vector3dVector(colors_class)
    # ウィンドウを作成してデータを表示
    o3d.visualization.draw_geometries([pcd_original, pcd_extracted,pcd_class])








