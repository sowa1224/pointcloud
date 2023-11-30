import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSVファイルのパス
csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_filtered.csv"

# CSVファイルの読み込み
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# 座標データを抽出
x = 1000 * data[:, 0]
y = 1000 * data[:, 1]
z = 1000 * data[:, 2]
nx = data[:, 6]
ny = data[:, 7]
nz = data[:, 8]


# 全点群データ
points = np.column_stack((x, y, z))

# 指定したy値
target_y = np.max(y)

# target_yに近い値を持つ点のインデックスを取得
indices = np.where(np.isclose(y, target_y))[0]

# 指定したx値に近い点のyとzの値を取得
target_x = x[indices]
target_z = z[indices]
target_nx = nx[indices]
target_ny = ny[indices]
target_nz = nz[indices]

# 頂点を設定
Top_x = target_x[0]
Top_y = np.max(y)
Top_z = target_z[0]
Top_direction=np.array([[target_nx[0],target_ny[0],target_nz[0]]])
Top_point = np.array([[Top_x, Top_y, Top_z]])

# 中心点のx座標を初期値として設定
center_x = Top_x

# 中心点に近い値を持つ点のインデックスを取得
indices_c = np.where(np.isclose(x, Top_x))[0]

# 指定したx値に近い点のyとzの値を取得
center_y = y[indices_c]
center_z = z[indices_c]
center_nx = nx[indices_c]
center_ny = ny[indices_c]
center_nz = nz[indices_c]

# 中心点（始点）を設定
Bot_x = Top_x
Bot_y = center_y[len(center_y) - 1]
Bot_z = center_z[len(center_z) - 1]
Bot_direction=np.array([[center_nx[len(center_nx) - 1],center_ny[len(center_ny) - 1],center_nz[len(center_nz) - 1]]])
Bot_point = np.array([[Bot_x, Bot_y, Bot_z]])

# からの配列を作成
extracted_points = np.empty((0, 3))
Final_x = Bot_x+9.0
   
# ひとつの点を格納する変数を初期化
extracted_point = None

while Bot_x <= Final_x:
    # 新しい始点のx座標
    new_bot_x = Bot_x + 4.5

    # 新しい始点に一番近い点を抽出
    selected_points = points[np.isclose(x, new_bot_x)]

    # 始点から頂点への方向ベクトルに近い条件を満たす点を抽出
    direction = selected_points - Bot_point
    dot_product = np.abs(np.dot(direction, Top_direction.T))
    threshold = 0.8
    selected_points = selected_points[dot_product <= threshold]

    # 最も近い点を抽出
    if len(selected_points) > 0:
        closest_point = selected_points[0]
        if extracted_point is None or np.linalg.norm(closest_point - Bot_point) < np.linalg.norm(extracted_point - Bot_point):
            extracted_point = closest_point
    # 始点に一番近い点を探す

    # 直線の方向ベクトルを計算
    direction = Top_point - extracted_point

    # 直線の方向ベクトルを正規化
    norm_direction = direction / np.linalg.norm(direction)

    # 始点と頂点の法線ベクトルを計算
    average_normal = np.mean(Bot_direction, extracted_point[6,9])

    # 各点と直線との距離を計算
    distances = np.abs(np.dot(points - extracted_point, norm_direction))

    # 直線に近い点の閾値
    threshold = 0.8

    # 法線ベクトルに近い点を抽出
    selected_points = points[distances <= threshold]
    selected_points = selected_points[np.abs(np.dot(selected_points - Top_point, average_normal)) <= angle_threshold]

    # 抽出した点をからの配列に追加
    extracted_points = np.append(extracted_points, selected_points, axis=0)

    # 新しい始点のx座標を次のループのために更新
    Bot_x = new_bot_x



"""""
print("抽出した点:")
print(extracted_points)

# 3Dプロットの準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 全点群データの表示（青色）
ax.scatter(x, y, z, c='blue', s=1)
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))

# 抽出した点の表示（赤色）
ax.scatter(extracted_points[:, 0], extracted_points[:, 1], extracted_points[:, 2], c='red', s=8)

# 軸ラベルの表示
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# グラフの表示
plt.show()

"""""
