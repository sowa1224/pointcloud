import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
import scipy.stats as stats


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

Final_x1 = Top_x+4.5
Final_x2 = Top_x-4.5

#ここからは近似曲線用のプログラム
# からの配列を作成
extracted_points = np.empty((0, 3))
extracted_points2 = np.empty((0, 3))

# 初期化
trajectory_count = 1  # 軌道の番号の初期値
trajectory_points = []  # 軌道ごとの点群データを保存するリスト


trajectory_count2 = 1  
trajectory_points2 = []  
    
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

    averaged_trajectory_points = []

    ####################################
    #分割する両
    split=15

    # 各軌道の点群データに対して処理を行う
    if len(selected_points) >= split:  # 点の数が20以上の場合に処理を行う
        # 点の数を分割数で割り、各分割の点数を計算
        split_count = len(selected_points) // split

        averaged_points = []
        # 分割数ごとに処理を行う
        for i in range(split):
            # 分割範囲のインデックスを計算
            start_index = i * split_count
            end_index = start_index + split_count

            # 分割範囲内の点の3次元座標を取得
            split_points = selected_points[start_index:end_index, :3]
            # 3次元座標の平均値を計算して新しいリストに追加
            averaged_point = np.mean(split_points, axis=0)
            averaged_points.append(averaged_point)

        averaged_trajectory_points.extend(averaged_points)


    # 軌道ごとの点群データをリストに追加
    trajectory_points.append(averaged_trajectory_points)


    # 抽出した点を表示するなどの処理を行う
    # 直線に近い点を抽出
    # 抽出した点をからの配列に追加
    extracted_points = np.append(extracted_points, averaged_trajectory_points, axis=0)

    # 選択した点のy座標がほとんど同じ場合、1つだけを残す
    #unique_indices = np.unique(extracted_points[:, 1], return_index=True)[1]
    #selected_points = extracted_points[unique_indices, :]

    # 抽出した点をからの配列に追加
    #extracted_points = np.append(extracted_points, selected_points, axis=0)

    print(f"extracの大きさは{extracted_points.shape}")


    # 軌道ごとの点群データをリストに追加
    trajectory_points.append(averaged_trajectory_points)

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

# 各軌道ごとの点群データの表示
for i, trajectory in trajectory_data.items():
    print(f"軌道{i}の点群データ:")
    print(trajectory)
    print(f"点の数: {len(trajectory)}")
   # print()

# 各軌道ごとの点群データの表示
for i, trajectory in trajectory_data2.items():
    print(f"軌道{i}の点群データ:")
    #print(trajectory)
    print(f"点の数: {len(trajectory)}")
    #print()

# extracted_pointsからx、y、z座標を取り出す
extracted_x = extracted_points[:, 0]
extracted_y = extracted_points[:, 1]
extracted_z = extracted_points[:, 2]

# extracted_points2からx、y、z座標を取り出す
extracted_x2 = extracted_points2[:, 0]
extracted_y2 = extracted_points2[:, 1]
extracted_z2 = extracted_points2[:, 2]

#print(extracted_points,extracted_points.shape)
#print(extracted_points2,extracted_points2.shape)


# 3Dプロットの準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 全点群データの表示（青色）
ax.scatter(x, y, z, c='blue', s=1)
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))

# 抽出した点の表示（赤色）
ax.scatter(extracted_x, extracted_y, extracted_z, c='red', s=8, label='Selected Points+')

# 抽出した点の表示（赤色）
#ax.scatter(extracted_x2, extracted_y2, extracted_z2, c='red', s=8, label='Selected Points-')

# 軸ラベルと凡例の表示

ax.set_ylabel('Y')
ax.set_zlabel('Z')


# グラフの表示
plt.show()




# 保存先のディレクトリパス
save_directory = "/home/sowa/prog/pointcloud/orbit/"


# 近似曲線を時間的な関数に変換するための関数
def time_function(t, start_point, end_point):
    # 始点と終点の座標を取得
    start_x, start_y, start_z = start_point
    end_x, end_y, end_z = end_point
    
    # 始点から終点までの距離を計算
    distance = np.linalg.norm(end_point - start_point)
    
    # 時間的な関数を定義
    def func(t):
        # 時間を0から1の範囲に正規化
        normalized_t = t / 10
        
        # 線形補間により点の座標を計算
        interpolated_x = start_x + (end_x - start_x) * normalized_t
        interpolated_y = start_y + (end_y - start_y) * normalized_t
        interpolated_z = start_z + (end_z - start_z) * normalized_t
        
        return np.array([interpolated_x, interpolated_y, interpolated_z])
    
    # 時間的な関数を適用して座標を計算
    points = np.apply_along_axis(func, axis=0, arr=t)
    
    return points


# 各軌道ごとに近似曲線を計算して保存
# フィッティングする関数モデルを定義
def fitting_func(x, a, b, c, d,e,f,g):
    return a * x[:, 0]**4 + b * x[:, 1]**4 + c * x[:, 0]**3  + d*x[:,1]**3+e * x[:, 0]**2  + f*x[:,1]**2+g

# 外れ値除去のための関数
def remove_outliers(data, threshold=3):
    z_scores = stats.zscore(data)  # Zスコアを計算
    filtered_data = data[np.abs(z_scores) < threshold]  # 閾値以下のデータを選択
    return filtered_data

# 各軌道ごとに近似曲線を計算して保存
for i, trajectory in trajectory_data.items():
    # 軌道の点群データからx座標、y座標、z座標を取得
    x = np.array(trajectory)[:, :2]
    z = np.array(trajectory)[:, 2]
    #x = trajectory[:, :2]
    #z = trajectory[:, 2]

    # 外れ値を除去
    filtered_z = remove_outliers(z)

    # フィッティングパラメータの初期値を設定
    initial_param = [3, 4, 4, 4,4,4,2]


    # 最小二乗法によるフィッティングを実行
    # フィッティングを実行
    optimized_param, _ = curve_fit(fitting_func, x, filtered_z, initial_param)

    # 近似曲線を計算
    approx_curve = fitting_func(x, *optimized_param)


    # 近似曲線を保存
    curve_save_path = save_directory + f"trajectory_{i}_curve.csv"
    np.savetxt(curve_save_path, np.column_stack((x[:, 0], x[:, 1], approx_curve)), delimiter=',', header='x,y,z', comments='')

    # 近似曲線パラメータを保存
    param_save_path = save_directory + f"trajectory_{i}_params.csv"
    np.savetxt(param_save_path, optimized_param, delimiter=',', header='a,b,c,d,e,f,g', comments='')

    # 元の点と近似曲線をプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x[:, 0], x[:, 1], z, color='b', label='Original Points')

    #ax.plot(x[:, 0], x[:, 1], approx_curve, color='r', label='Approximated Curve')
    ax.plot(x[:, 0], x[:, 1], approx_curve, color='r')

    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    # 軸の目盛りの設定
    ax.tick_params(axis='x', which='both', labelsize=12, width=20, length=10)  # X軸の目盛り
    ax.tick_params(axis='y', which='both', labelsize=12, width=20, length=10)  # Y軸の目盛り
    ax.tick_params(axis='z', which='both', labelsize=12, width=20, length=30)  # Z軸の目盛り

    ax.legend()
   # plt.title(f"Trajectory {i}")
    plt.show()

    print(f"軌道{i}の近似曲線を保存しました: {curve_save_path}")
    print(f"軌道{i}の近似曲線パラメータを保存しました: {param_save_path}")



"""""
# 近似曲線のパラメータを読み込む
param_file_path = "/home/sowa/prog/pointcloud/orbit/trajectory_1_params.csv"
# 近似曲線のパラメータを読み込む
optimized_param = np.loadtxt(param_file_path, delimiter=',', skiprows=1)


# 近似曲線の始点と終点を設定
start_point = np.array([-6.604000000000000625e+01,6.536870000000000402e+01,-3.192976827638303803e+02])
end_point = np.array([-6.195069999999999766e+01,-8.142090000000000316e+01,-3.031669204468375938e+02])

# 時間的な関数を計算
t = np.linspace(0, 10, num=30)
time_points = time_function(t, start_point, end_point)

# 時間的な関数をプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(time_points[:, 0], time_points[:, 1], time_points[:, 2], color='b', label='Time-based Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Time-based Function')
plt.show()

"""""










