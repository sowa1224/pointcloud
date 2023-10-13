import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

# CSVファイルのパス
csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_filtered.csv"

#matplotlib座標軸で表示
# CSVファイルのパス
# CSVファイルの読み込み
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

#ここはmm単位での選定
# 座標データを抽出
x = 1000 * data[:, 0]
y = 1000 * data[:, 1]
z = 1000 * data[:, 2]
#0が縦軸　1が垂直　2が横軸

#全点群データ
points = np.column_stack((x, y, z))
print(points)
print(points.shape)

#ここからは座標設計ように点群の選定を行う、
#案①：y軸最大最小を頂点、始点と設定して、始点から範囲内の点群の内、点aと始点との単位ベクトル、点aと終点との単位ベクトルの内積が最も1に近い点を選択
#案②：①と同じ始点と頂点、ただし単位ベクトルの内積は始点と選択点との単ベと始点と頂点のとの単ベの積が１に近い方
#案③：同じ始点と頂点が設定されて、二点でつなげた直線上にある点もしくは仮想点を選択/設定、つまり分水嶺を作る

# 指定したy値
target_y = np.max(y)
print(f"yの最大値は{target_y}")
# targetxに近い値を持つ点のインデックスを取得
indices = np.where(np.isclose(y, target_y))[0]
print(indices)
# 指定したx値に近い点のyとzの値を取得
target_x = x[indices]
target_z = z[indices]

#頂点を設定
Top_x = target_x[0]
Top_y = np.max(y)
Top_z = target_z[0]
Top_point = np.array([[Top_x, Top_y, Top_z]])
print(f"頂点は({Top_x}, {Top_y}, {Top_z})")
print(Top_point.shape)

# 中心点のx座標を初期値として設定
center_x = Top_x
# 中心点に近い値を持つ点のインデックスを取得
indices_c = np.where(np.isclose(x, Top_x))[0]

# 指定したx値に近い点のyとzの値を取得
center_y = y[indices_c]
center_z = z[indices_c]
print(center_y)
print(center_z)

#中心点（始点）を設定
Bot_x = Top_x
Bot_y = center_y[len(center_y)-1]
Bot_z = center_z[len(center_z)-1]
Bot_point = np.array([[Bot_x, Bot_y, Bot_z]])
print(f"始点は({Bot_x}, {Bot_y}, {Bot_z})")
print(Bot_point.shape)

# からの配列を作成
extracted_points = np.empty((0, 3))
extracted_points2 = np.empty((0, 3))

# 初期化
trajectory_count = 1  # 軌道の番号の初期値
trajectory_points = []  # 軌道ごとの点群データを保存するリスト

trajectory_count2 = 1  
trajectory_points2 = []  

Final_x1 = Top_x+4.5
Final_x2 = Top_x-4.5

#ここからは近似曲線用のプログラム
# からの配列を作成
extracted_points = np.empty((0, 3))
extracted_points2 = np.empty((0, 3))

   
while Bot_x <= Final_x1:
    # 新しい始点のx座標
    new_bot_x = Bot_x + 4.5

    #始点に一番近い点を探す

    # 直線の方向ベクトルを計算
    direction = Top_point[0, :2] - np.array([new_bot_x, Bot_y])

    # 直線の方向ベクトルを正規化
    norm_direction = direction / np.linalg.norm(direction)

    # 各点と直線との距離を計算
    distances = np.abs(np.cross(points[:, :2] - np.array([new_bot_x, Bot_y]), norm_direction))

    # 直線に近い点の閾値
    threshold = 0.8

    # 直線に近い点を抽出
    selected_points = points[distances <= threshold]

    # 抽出した点を表示するなどの処理を行う
    # 直線に近い点を抽出
    # 抽出した点をからの配列に追加
    extracted_points = np.append(extracted_points, selected_points, axis=0)

    # 軌道ごとの点群データをリストに追加
    trajectory_points.append(selected_points)

    # 新しい始点のx座標を次のループのために更新
    Bot_x = new_bot_x

    # 軌道番号を更新
    trajectory_count += 1


while Bot_x >= Final_x2:
    # 新しい始点のx座標
    new_bot_x = Bot_x - 4.5

    # 直線の方向ベクトルを計算
    direction = Top_point[0, :2] - np.array([new_bot_x, Bot_y])

    # 直線の方向ベクトルを正規化
    norm_direction = direction / np.linalg.norm(direction)

    # 各点と直線との距離を計算
    distances = np.abs(np.cross(points[:, :2] - np.array([new_bot_x, Bot_y]), norm_direction))

    # 直線に近い点の閾値
    threshold = 0.5

    # 直線に近い点を抽出
    selected_points2 = points[distances <= threshold]

    # 抽出した点を表示するなどの処理を行う
    # 直線に近い点を抽出
     # 抽出した点をからの配列に追加
    extracted_points2 = np.append(extracted_points2, selected_points2, axis=0)

    # 軌道ごとの点群データをリストに追加
    trajectory_points2.append(selected_points2)

    # 新しい始点のx座標を次のループのために更新
    Bot_x = new_bot_x

    # 軌道番号を更新
    trajectory_count2 += 1

    Bot_x = new_bot_x

# 各軌道ごとに番号を付けるための配列を作成
trajectory_indices = np.arange(1, trajectory_count)
trajectory_indices2 = np.arange(1, trajectory_count2)

# 各軌道ごとの点群データに番号を付けて保存
trajectory_data = dict(zip(trajectory_indices, trajectory_points))
trajectory_data2 = dict(zip(trajectory_indices2, trajectory_points2))


print(f"大きさは{len(trajectory_points)}")


"""""
# スプライン曲線の次数を設定
degree = 3  # 3次のスプライン曲線を使用


# 軌道1のスプライン曲線をフィッティング
trajectory1_points = np.array(trajectory_points)  # 軌道1の点群データをNumPy配列に変換
t1 = np.arange(trajectory1_points.shape[0])  # パラメータtを設定（点のインデックスを使用）
spl1 = CubicSpline(t1, trajectory1_points.T, bc_type='clamped')  # スプライン曲線のフィッティング

# 軌道2のスプライン曲線をフィッティング
trajectory2_points = np.array(trajectory_points2)  # 軌道2の点群データをNumPy配列に変換
t2 = np.arange(trajectory2_points.shape[0])  # パラメータtを設定（点のインデックスを使用）
spl2 = CubicSpline(t2, trajectory2_points.T, bc_type='clamped')  # スプライン曲線のフィッティング

# スプライン曲線上の点を生成
t_new = np.linspace(t1[0], t1[-1], 100)  # スプライン曲線上のパラメータtを生成
trajectory1_smooth = spl1(t_new)  # 軌道1のスプライン曲線上の点を計算
trajectory2_smooth = spl2(t_new)  # 軌道2のスプライン曲線上の点を計算

# 可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 全点群データの表示（青色）
ax.scatter(x, y, z, c='blue', s=1)

# 軌道1のスプライン曲線を表示（赤色）
ax.plot(trajectory1_smooth[:, 0], trajectory1_smooth[:, 1], trajectory1_smooth[:, 2], c='red', label='Trajectory 1')

# 軌道2のスプライン曲線を表示（緑色）
ax.plot(trajectory2_smooth[:, 0], trajectory2_smooth[:, 1], trajectory2_smooth[:, 2], c='green', label='Trajectory 2')

# 軸ラベルと凡例の表示
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# グラフの表示
plt.show()

"""""