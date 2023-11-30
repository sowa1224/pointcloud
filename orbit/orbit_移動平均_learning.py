import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
import scipy.stats as stats


csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_filtered.csv"
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

#ここはmm単位での選定
# 座標データを抽出
x = 1000 * data[:, 0]
y = 1000 * data[:, 1]
z = 1000 * data[:, 2]
#0が縦軸　1が垂直　2が横軸

#全点群データ
points = np.column_stack((x, y, z))
#print(points)
#print(points.shape)

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
#print(center_y)
#print(center_z)

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
trajectory_count = 1  # 軌道の番号の初期値
trajectory_points = []  # 軌道ごとの点群データを保存するリスト


window_size = int(input("移動平均ウィンドウサイズ："))

def calculate_moving_average(data, window_size):
    # ウィンドウサイズが奇数でない場合、奇数になるように調整
    if window_size % 2 == 0:
        window_size += 1

    # ウィンドウサイズの半分を計算
    half_window = window_size // 2

    # 入力データから始点と終点を除外
    filtered_data = data[1:-1]

    # 移動平均を計算するための空の配列を作成
    smoothed_data = np.copy(filtered_data)

    # 移動平均を計算
    for i in range(half_window, len(filtered_data) - half_window):
        window = filtered_data[i - half_window:i + half_window + 1]
        smoothed_data[i - half_window] = np.mean(window, axis=0)

    # 始点と終点を再度追加
    smoothed_data = np.vstack((data[0], smoothed_data, data[-1]))

    return smoothed_data


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

    print("selected_points:")
    print(selected_points)

    # selected_pointsから始点と終点を除外
    #filtered_points = selected_points[1:-1]

    # 各列に対して移動平均を計算
    #smoothed_points = np.copy(filtered_points)  # 平滑化された点群を保存するための配列


    #for i in range(3):  # インデックス0から2までの範囲でループ
     #   column_data = filtered_points[:, i]  # 各列のデータを取得
      #  smoothed_data = np.convolve(column_data, np.ones(window_size) / window_size, mode='same')  # 移動平均を計算
       # smoothed_points[:, i] = smoothed_data

    # 始点と終点を含む平滑化された点を作成
    #smoothed_selected_points = np.vstack((selected_points[:1], smoothed_points, selected_points[-1:]))
    smoothed_points = calculate_moving_average(selected_points, window_size)
    extracted_points = np.append(extracted_points,smoothed_points, axis=0)

    print(f"extracの大きさは{extracted_points.shape}")

    # 軌道ごとの点群データをリストに追加
    trajectory_points.append(smoothed_points)

    # 新しい始点のx座標を次のループのために更新
    Bot_x = new_bot_x

    # 軌道番号を更新
    trajectory_count += 1

# 各軌道ごとに番号を付けるための配列を作成
    trajectory_indices = np.arange(1, trajectory_count)

# 各軌道ごとの点群データに番号を付けて保存
    trajectory_data = dict(zip(trajectory_indices, trajectory_points))





# 各軌道ごとの点群データの表示
for i, trajectory in trajectory_data.items():
    print(f"軌道{i}の点群データ:")
    print(trajectory)
    print(f"点の数: {len(trajectory)}")
extracted_x = extracted_points[:, 0]
extracted_y = extracted_points[:, 1]
extracted_z = extracted_points[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', s=1)
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))
ax.scatter(extracted_x, extracted_y, extracted_z, c='red', s=8, label='Selected Points+')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
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
    curve_save_path = save_directory + f"trajectory_{i}_curve_移動平均.csv"
    np.savetxt(curve_save_path, np.column_stack((x[:, 0], x[:, 1], approx_curve)), delimiter=',', header='x,y,z', comments='')

    # 近似曲線パラメータを保存
    param_save_path = save_directory + f"trajectory_{i}_params_移動平均.csv"
    np.savetxt(param_save_path, optimized_param, delimiter=',', header='a,b,c,d,e,f,g', comments='')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x[:, 0], x[:, 1], z, color='b', label='Original Points')

    #ax.plot(x[:, 0], x[:, 1], approx_curve, color='r', label='Approximated Curve')
    ax.plot(x[:, 0], x[:, 1], approx_curve, color='b')

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



