import numpy as np
from scipy.interpolate import griddata
import open3d as o3d

# CSVファイルのパス
csv_file_path = "/home/sowa/prog/pointcloud/csv/carrot_filtered.csv"

# CSVファイルの読み込み
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# 座標データを抽出
x = 1000 * data[:, 0]
y = 1000 * data[:, 1]
z = 1000 * data[:, 2]

# データの欠損値処理
valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
x_valid = x[valid_mask]
y_valid = y[valid_mask]
z_valid = z[valid_mask]

# 格子点の範囲を調整して逆方向に生成
xi = np.linspace(max(x_valid), min(x_valid), 100)  # x座標の範囲を逆に設定
yi = np.linspace(max(y_valid), min(y_valid), 100)  # y座標の範囲を逆に設定
xi, yi = np.meshgrid(xi, yi)

# 近似曲面を計算する（補間方法を変更）
zi = griddata((x_valid, y_valid), z_valid, (xi, yi), method='cubic')

# 三角メッシュを生成
mesh = o3d.geometry.TriangleMesh()
vertices = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()]).T
mesh.vertices = o3d.utility.Vector3dVector(vertices)

# 頂点法線を計算
mesh.compute_vertex_normals()

# 三角メッシュを可視化
o3d.visualization.draw_geometries([mesh])
