import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# CSVファイルのパス
#csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_filtered.csv"
csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_filtered_1011.csv"
# CSVファイルを読み込む
df = pd.read_csv(csv_file_path)

# 座標データを抽出
points = df[['X', 'Y', 'Z']].values

# 色データを抽出
colors = df[['R', 'G', 'B']].values # 0-255の範囲から0-1の範囲に変換

# 法線データを抽出
normals = df[['nX', 'nY', 'nZ']].values 

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 可視化
o3d.visualization.draw_geometries([pcd])



#matplotlib座標軸で表示
# CSVファイルのパス
# CSVファイルの読み込み
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

#ここはmm単位での選定
# 座標データを抽出
x = 1000*data[:, 0]
y = 1000*data[:, 1]
z = 1000*data[:, 2]
#0が縦軸　1が垂直　2が横軸

#3Dグラフの作成
fig = plt.figure(figsize=(8, 6)) # 幅8インチ、高さ6インチの図を作成
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c='b', marker='o')

# y軸の表示範囲を調整,この場合y軸はxとｚ軸の②倍になる
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))

#軸やラベルの追加
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#可視化
plt.show()



""""

#ここからは座標設計ように点群の選定を行う、
#案①：y軸最大最小を頂点、始点と設定して、始点から範囲内の点群の内、点aと始点との単位ベクトル、点aと終点との単位ベクトルの内積が最も1に近い点を選択
#案②：①と同じ始点と頂点、ただし単位ベクトルの内積は始点と選択点との単ベと始点と頂点のとの単ベの積が１に近い方
#案③：同じ始点と頂点が設定されて、二点でつなげた直線上にある点もしくは仮想点を選択/設定、つまり分水嶺を作る

#全点群データ
points = np.array([x,y,z])
print(points)
# 指定したy値
target_y = (np.max(y))
print(f"yの最大値は{target_y}")
# targetxに近い値を持つ点のインデックスを取得
indices = np.where(np.isclose(y, target_y))[0]
print(indices)
# 指定したx値に近い点のyとzの値を取得
target_x =x[indices]
target_z = z[indices]

#頂点を設定
Top_x= target_x[0]
Top_y= np.max(y)
Top_z= target_z[0]
Top_point=np.array([Top_x,Top_y,Top_z])
print(f"頂点は({Top_x},{Top_y},{Top_z})")

# 中心点のx座標を初期値として設定
center_y = np.min(y)
# 中心点に近い値を持つ点のインデックスを取得
indices_c = np.where(np.isclose(y, center_y))[0]
print(indices_c)
# 指定したx値に近い点のyとzの値を取得
center_x =x[indices_c]
center_z = z[indices_c]

#中心点（始点）を設定
Bot_x= center_x[0]
Bot_y= np.min(y)
Bot_z= center_z[0]
Bot_point=np.array([Bot_x,Bot_y,Bot_z])
print(f"始点は({Bot_x},{Bot_y},{Bot_z})")

fig=plt.figure()
ax= fig.add_subplot(111,projection="3d")
ax.scatter(x, y, z, color='blue') 
ax.scatter(Top_x,Top_y,Top_z,marker="o",color="red")
ax.scatter(Bot_x,Bot_y,Bot_z,marker="o",color="red")
plt.show()


#半径内の点群を抽出し、かつその２つの点のX-Y面上の単位ベクトル
#軌道用の点群を格納
Orbit_1 = []


# 始点から頂点への方向ベクトル
line_direction = Top_point-Bot_point

# 点群を2次元に変換
points_2d = points[:, :2]

# 各点と直線とのなす角のcosineを計算
cosines = np.dot(points_2d - Top_point[:2], np.tile(line_direction[:2], (len(points_2d), 1))) / (np.linalg.norm(points_2d - Top_point[:2], axis=1) * np.linalg.norm(line_direction[:2]))

# 直線に近い点を抽出
threshold = 0.9  # 直線abに近い点を選択するための閾値
selected_points = points[np.abs(cosines) >= threshold]


# 空のリストを作成して選択された点を保存する
selected_points = []
points = np.column_stack((x, y, z))
print(points)

point_a = np.array([Bot_x,Bot_y,Bot_z])


# 直線abの単位ベクトルを計算
line_direction = (Top_point - point_a) / np.linalg.norm(Top_point - point_a)

# 各点と直線abとのなす角のcosineを計算
cosines = np.dot(points - point_a, line_direction) / (np.linalg.norm(points - point_a, axis=1) * np.linalg.norm(line_direction))

# 90度に近い点のインデックスを取得
indices = np.where(np.isclose(np.abs(cosines), 1, atol=0.1))[0]

# 選択された点を配列に追加
selected_points.extend(points[indices])


# 点aを中心に半径2以内の点の選択を繰り返すループ
while True:

    # 中心点から半径2以内の点を抽出
    mask = np.linalg.norm(points - point_a, axis=1) <= 2.0
    filtered_points = points[mask]

    if len(filtered_points) == 0:
        break

 # 各点と始点との単位ベクトルを計算
    norm_values = np.linalg.norm(filtered_points - point_a, axis=1)
    norm_values[norm_values == 0] = np.finfo(float).eps  # ゼロ除算を避けるための処理
    unit_vectors_a = (filtered_points - point_a) / norm_values[:, np.newaxis]


    # 各点と終点との単位ベクトルを計算
    unit_vectors_b = (Top_point - filtered_points) / np.linalg.norm(Top_point - filtered_points, axis=1)[:, np.newaxis]

    # 各点との内積を計算
    dot_products = np.sum(unit_vectors_a * unit_vectors_b, axis=1)

    # 最も1に近い内積の点を選択
    closest_point_index = np.argmax(dot_products)
    closest_point = filtered_points[closest_point_index]

    # 選択された点のy座標が点bのy座標に近い場合、ループを終了
    if np.isclose(closest_point[1], Top_point[1]):
        break

    # 選択された点をリストに追加
    selected_points.append(closest_point)

    # 選択された点を次の中心点として更新
    point_a = closest_point

    """""
