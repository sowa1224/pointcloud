#ノート
# 白い点群データを抽出
#にんじんの点群を抽出するプログラムの順番：
#ply_to_csv→csv_hsv_extract→csv_RGB_extrac→csv_show_open3d_matlab
#点群ファイル　plyファイルを得る手法として　realsense -sdkによる手動　もしくは/home/sowa/prog/librealsense-2.54.1/wrappers/python/examples　でのexport...example2.plyのファイルを実行すると点群を得られる


#2023/9/12
#下のコードは移動平均プログラムを表示と近似曲線を描く用のプログラム
#現在まだ移動平均ができているかどうかの確認をしている、できたら下を進む
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
    curve_save_path = save_directory + f"trajectory_{i}_curve_移動平均.csv"
    np.savetxt(curve_save_path, np.column_stack((x[:, 0], x[:, 1], approx_curve)), delimiter=',', header='x,y,z', comments='')

    # 近似曲線パラメータを保存
    param_save_path = save_directory + f"trajectory_{i}_params_移動平均.csv"
    np.savetxt(param_save_path, optimized_param, delimiter=',', header='a,b,c,d,e,f,g', comments='')

    # 元の点と近似曲線をプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], z, color='b', label='Original Points')

    ax.plot(x[:, 0], x[:, 1], approx_curve, color='r', label='Approximated Curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title(f"Trajectory {i}")
    plt.show()

    print(f"軌道{i}の近似曲線を保存しました: {curve_save_path}")
    print(f"軌道{i}の近似曲線パラメータを保存しました: {param_save_path}")