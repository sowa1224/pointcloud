import pandas as pd
import numpy as np
import open3d as o3d
# CSVファイルの読み込み
#csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_cr.csv"
csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_1011.csv"

data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']

#data = pd.read_csv(csv_file_path)

#csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot.csv"
#df = pd.read_csv(csv_file_path, header=None)
#df.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']


#もしCSVファイルにヘッダーがない場合、いかに変える
#data = pd.read_csv(csv_file_path, header=None) # ヘッダー行をスキップ
#data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']



# RGBデータを抽出
colors = data[['R', 'G', 'B']]

# 原点付近の50個のRGBデータを抽出
selected_colors = colors.head(500)

# RGB平均値を計算
rgb_mean = selected_colors.mean()

# 結果を表示
print("RGB平均値:", rgb_mean)


# RGBの最小値と最大値を定義
#カメラの原点に野菜の中心があるとして、その中心付近のRGBを参考に点群を抽出
#i=100 #上下限
#R_min = rgb_mean[0]*255-i
#R_max = rgb_mean[0]*255+i
#G_min = rgb_mean[1]*255-i
#G_max = rgb_mean[1]*255+i
#B_min = rgb_mean[2]*255-i
#B_max = rgb_mean[2]*255+i

# RGBの最小値と最大値を定義
# にんじんのRGB R:244 G:122 B:68
#i=90 #上下限
#R=244
#G=122
#B=68
#R_min = R-i
#R_max = R+i
#G_min = G-i
#G_max = G+i
#B_min = B-i
#B_max = B+i

# じゃがいものRGB R:245 G:190 B:136
i=30 #上下限
R=190
G=160
B=100
R_min = R-50
R_max = R+80
G_min = G-50
G_max = G+50
B_min = B-40
B_max = B+40


# RGBデータを参考に、対象物の点群を抽出
carrot_data = data[(data['R'] > R_min/255) & (data['R'] < R_max/255) & (data['G'] > G_min/255) & (data['G'] < G_max/255) & (data['B'] > B_min/255) & (data['B'] < B_max/255)]

# CSVファイルのパス
save_path = "/home/sowa/prog/pointcloud/csv/potato_extract_RGB.csv"
#save_path = "/home/sowa/prog/pointcloud/csv/carrot_extract_RGB.csv"



# 新しいCSVファイルとして保存
carrot_data.to_csv(save_path, index=False)

# 保存されたCSVファイルを読み込む
df = pd.read_csv(save_path)

# 座標データを抽出
points = df[['X', 'Y', 'Z']].values

# 色データを抽出
colors = df[['R', 'G', 'B']].values  # 0-255の範囲から0-1の範囲に変換

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可視化
o3d.visualization.draw_geometries([pcd])


#次に半径外れ値除去で更にサンプリングする
# 抽出済みの点群データを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_extract_RGB.csv"
#csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_extract_RGB.csv"
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# 座標データを抽出
points = data[:, :3]

# RGBデータを抽出
colors = data[:, 3:6] # 0-255の範囲から0-1の範囲に変換

# 法線データを抽出
normals = data[:, 6:]

# 半径外れ値除去法のパラメータを設定
radius = 0.007 # 半径の閾値
min_points = 60  # 最小点数の閾値

# Open3DのPointCloudオブジェクトを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)
#もしpcd.clorsがなければ、可視化される点群データは色は含まれる

# 半径外れ値除去法を適用
pcd_filtered, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

# 可視化
o3d.visualization.draw_geometries([pcd_filtered])

# フィルタリング後の座標データを取得
filtered_points = np.asarray(pcd_filtered.points)

# フィルタリング後のRGBデータを取得
filtered_colors = np.asarray(pcd_filtered.colors)

# フィルタリング後の法線データを取得
filtered_normals = np.asarray(pcd_filtered.normals)

# フィルタリング後のデータを結合
filtered_data = np.concatenate((filtered_points, filtered_colors, filtered_normals), axis=1)

# 新しいCSVファイルとして保存
save_path = "/home/sowa/prog/pointcloud/csv/potato_filtered.csv"
#save_path = "/home/sowa/prog/pointcloud/csv/carrot_filtered.csv"
np.savetxt(save_path, filtered_data, delimiter=',', header='X,Y,Z,R,G,B,nX,nY,nZ', comments='')


