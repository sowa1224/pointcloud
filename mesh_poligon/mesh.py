import numpy as np
from scipy.interpolate import griddata
import open3d as o3d
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

# データの欠損値処理
valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
x_valid = x[valid_mask]
y_valid = y[valid_mask]
z_valid = z[valid_mask]

print(max(x_valid),min(x_valid))
print(max(y_valid),min(y_valid))

# 格子点の範囲を調整

# 格子点の範囲を調整して逆方向に生成
xi = np.linspace(max(x_valid), min(x_valid), 100)  # x座標の範囲を逆に設定
yi = np.linspace(max(y_valid), min(y_valid), 100)  # y座標の範囲を逆に設定
xi, yi = np.meshgrid(xi, yi)


""""
# 格子点の範囲を調整
xi = np.linspace(min(x_valid), max(x_valid), 100)  # x座標の範囲を適宜調整する
yi = np.linspace(min(y_valid), max(y_valid), 100)  # y座標の範囲を適宜調整する
xi, yi = np.meshgrid(xi, yi)
"""
# 近似曲面を計算する（補間方法を変更）
zi = griddata((x_valid, y_valid), z_valid, (xi, yi), method='cubic')

# 近似曲面上の点を保存するための空の配列を作成
approx_surface_points = []

# 近似曲面上の点を抽出し、空の配列に保存
for i in range(len(xi)):
    for j in range(len(yi)):
        approx_surface_points.append([xi[i, j], yi[i, j], zi[i, j]])

# 空の配列をNumPy配列に変換
approx_surface_points = np.array(approx_surface_points)

# 空の配列に保存された近似曲面上の点を表示
print(approx_surface_points)
print(approx_surface_points.shape)

# NaN値を除外する
valid_mask = ~np.isnan(zi)
zi_valid = zi[valid_mask]
xi_valid = xi[valid_mask]
yi_valid = yi[valid_mask]

# 近似曲面の座標データを作成
surface_points = np.column_stack((xi_valid.flatten(), yi_valid.flatten(), zi_valid.flatten()))

# Open3Dでの可視化
surface_pcd = o3d.geometry.PointCloud()
surface_pcd.points = o3d.utility.Vector3dVector(surface_points)
o3d.visualization.draw_geometries([surface_pcd])



# Matplotlibでの可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 格子点座標のメッシュグリッドを作成
xi_mesh, yi_mesh = np.meshgrid(xi_valid, yi_valid)

# zi_valid の形状をメッシュグリッドに合わせるため、新しい2次元のグリッドデータを作成
zi_valid_mesh = np.zeros(xi_mesh.shape)
for i in range(len(xi_valid)):
    for j in range(len(yi_valid)):
        zi_valid_mesh[j, i] = zi_valid[i * len(yi_valid) + j]

# 近似曲面の3Dプロット
ax.plot_surface(xi_mesh, yi_mesh, zi_valid_mesh, cmap='viridis')


# 軸ラベルの設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# グラフの表示
plt.show()
