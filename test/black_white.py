
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance


csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1017.csv"
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

# 新しいCSVファイルとして保存
#1番目のpathの方は芽が２つ　２番めのpathは４つある
output_csv_path = "/home/sowa/prog/pointcloud/csv/extracted_points_and_original.csv"
#output_csv_path = "/home/sowa/prog/pointcloud/csv/extracted_data1025.csv"

# 抽出用の点群を黒、それ以外を白に設定
extracted_xyzrgb = extracted_data[['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']].values
other_xyzrgb = df[~df.index.isin(extracted_data.index)][['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']].values

# 色の設定
black_color = [0, 0, 0]  # 黒
white_color = [255, 255, 255]  # 白

# 抽出用の点を黒、それ以外を白に設定
for point in extracted_xyzrgb:
    point[3:6] = black_color  # RGBを黒に設定

for point in other_xyzrgb:
    point[3:6] = white_color  # RGBを白に設定

# 新しいCSVファイルとして保存
output_csv_path = "/home/sowa/prog/pointcloud/csv/combined_points.csv"

# 抽出用の点群を黒、それ以外を白に設定
combined_xyzrgb = np.vstack((extracted_xyzrgb, other_xyzrgb))

# 新しいデータフレームを作成
combined_df = pd.DataFrame(data=combined_xyzrgb, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ'])

# 新しいデータフレームをCSVファイルとして保存
combined_df.to_csv(output_csv_path, index=False)

# CSVファイルからデータを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/combined_points.csv"
df = pd.read_csv(csv_file_path)

# 3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# データをプロット
ax.scatter(df['X'], df['Y'], df['Z'], c=df[['R', 'G', 'B']] / 255.0, marker='o')

# 軸ラベルの設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# グラフの表示
plt.show()

print(extracted_xyzrgb)
print(extracted_xyzrgb.shape) #このコードでは9x9


# サンプルの点群データ（各点の(x, y)座標）
point_cloud = extracted_xyzrgb[:,:3]
print(point_cloud)

# 探索距離の閾値
distance_threshold = 0.02

# ポイント間の距離行列を計算
distance_matrix = distance.cdist(point_cloud, point_cloud, 'euclidean')

print(distance_matrix)
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
for i, cluster in enumerate(clusters):
    print(f"クラスタ {i + 1}: {point_cloud[cluster]}")