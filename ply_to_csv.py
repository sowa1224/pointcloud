import open3d as o3d
import numpy as np
import pandas as pd

# PLYファイルのパス
#ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/carrot2.ply"
ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1011.ply"
# PLYファイルの読み込み
ptCloud = o3d.io.read_point_cloud(ply_file)

# 点群データをnumpy配列として取得
points = np.asarray(ptCloud.points)
colors = np.asarray(ptCloud.colors)
normals = np.asarray(ptCloud.normals)

# 深度が5メートル以内かつ法線ベクトルが[0, 0, 1]以外の点群を抽出
selected_points = points
selected_colors = colors
selected_normals = normals

# 抽出した点群データをDataFrameに変換
df = pd.DataFrame({
    'X': selected_points[:, 0],
    'Y': selected_points[:, 1],
    'Z': selected_points[:, 2],
    'Red': selected_colors[:, 0],
    'Green': selected_colors[:, 1],
    'Blue': selected_colors[:, 2],
    'nX': selected_normals[:, 0],
    'nY': selected_normals[:, 1],
    'nZ': selected_normals[:, 2]
})

# CSVファイルとして保存（ヘッダー行を含めずに保存）
#headerをFalseにすることで、ヘッダーをなくす
#df.to_csv('/home/sowa/prog/pointcloud/csv/carrot_test.csv', index=False, header=False)
df.to_csv('/home/sowa/prog/pointcloud/csv/potato_viewer_1011.csv', index=False, header=False)
print("点群データがCSVファイルに保存されました。")
print("CSVファイルの大きさは",df.shape)


