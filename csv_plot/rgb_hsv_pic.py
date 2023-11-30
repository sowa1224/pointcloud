import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import rgb_to_hsv

file_paths = [
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0201_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0202_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0203_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0204_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0205_filtered.csv',
]

fig, axes = plt.subplots(5, 4, figsize=(15, 20))  # 5つのファイルに対して5行4列のサブプロットを作成

for idx, path in enumerate(file_paths):
    data = pd.read_csv(path, header=None)
    data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    coordinates = data[['X', 'Y', 'Z']].values
    rgb_data = data[['R', 'G', 'B']].values / 255.0
    hsv_data = rgb_to_hsv(rgb_data)

    hue_data = hsv_data[:, 0] * 255
    saturation_data = hsv_data[:, 1] * 255
    value_data = hsv_data[:, 2] * 255

    num_points = data.shape[0]

    hue_image, saturation_image, value_image = hue_data.reshape((1, num_points)), saturation_data.reshape(
        (1, num_points)), value_data.reshape((1, num_points))

    row = idx  # 現在のインデックスに対応する行
    col = 0    # 列のインデックス

    scatter_hue = axes[row, col].scatter(coordinates[:, 0], coordinates[:, 1], c=hue_data, cmap='hsv', s=5)
    #axes[row, col].set_title(f'Hue Image for {path}')
    axes[row, col].set_xlabel('X')
    axes[row, col].set_ylabel('Y')

    scatter_saturation = axes[row, col + 1].scatter(coordinates[:, 0], coordinates[:, 1], c=saturation_data, cmap='hsv', s=5)
    #axes[row, col + 1].set_title(f'Saturation Image for {path}')
    axes[row, col + 1].set_xlabel('X')
    axes[row, col + 1].set_ylabel('Y')

    scatter_value = axes[row, col + 2].scatter(coordinates[:, 0], coordinates[:, 1], c=value_data, cmap='hsv', s=5)
    #axes[row, col + 2].set_title(f'Value Image for {path}')
    axes[row, col + 2].set_xlabel('X')
    axes[row, col + 2].set_ylabel('Y')

    scatter_rgb = axes[row, col + 3].scatter(coordinates[:, 0], coordinates[:, 1], c=rgb_data, s=5)
    #axes[row, col + 3].set_title(f'Colored 2D Point Cloud for {path}')
    axes[row, col + 3].set_xlabel('X')
    axes[row, col + 3].set_ylabel('Y')

    fig.colorbar(scatter_hue, ax=axes[row, col], orientation='vertical', label='Hue')
    fig.colorbar(scatter_saturation, ax=axes[row, col + 1], orientation='vertical', label='Saturation')
    fig.colorbar(scatter_value, ax=axes[row, col + 2], orientation='vertical', label='Value')
    fig.colorbar(scatter_rgb, ax=axes[row, col + 3], orientation='vertical', label='RGB')

plt.tight_layout()
plt.show()


"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
import os

csv_file_path ='/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0209_filtered.csv'
data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

# 座標とRGBデータを取り出す
coordinates = data[['X', 'Y', 'Z']].values
rgb_data = data[['R', 'G', 'B']].values / 255.0  # RGBデータを0から1の範囲に正規化

# RGBからHSVに変換
hsv_data = rgb_to_hsv(rgb_data)

# 色相, 彩度, 明度データを取り出す
hue_data = hsv_data[:, 0]*255
saturation_data = hsv_data[:, 1]*255
value_data = hsv_data[:, 2]*255

# サイズを取得
num_points = data.shape[0]

# 色相, 彩度, 明度データをreshape
hue_image, saturation_image, value_image = hue_data.reshape((1, num_points)), saturation_data.reshape((1, num_points)), value_data.reshape((1, num_points))

# サブプロットを作成
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 色相画像をプロット
scatter_hue = axes[0, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=hue_data, cmap='hsv', s=5)
axes[0, 0].set_title('Hue Image for Point Cloud')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# 彩度画像をプロット
scatter_saturation = axes[0, 1].scatter(coordinates[:, 0], coordinates[:, 1], c=saturation_data, cmap='hsv', s=5)
axes[0, 1].set_title('Saturation Image for Point Cloud')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')

# 明度画像をプロット
scatter_value = axes[1, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=value_data, cmap='hsv', s=5)
axes[1, 0].set_title('Value Image for Point Cloud')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')

# RGB画像をプロット
scatter_rgb = axes[1, 1].scatter(coordinates[:, 0], coordinates[:, 1], c=rgb_data, s=5)
axes[1, 1].set_title('Colored 2D Point Cloud')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')

# カラーバーを追加
fig.colorbar(scatter_hue, ax=axes[0, 0], orientation='vertical', label='Hue')
fig.colorbar(scatter_saturation, ax=axes[0, 1], orientation='vertical', label='Saturation')
fig.colorbar(scatter_value, ax=axes[1, 0], orientation='vertical', label='Value')
fig.colorbar(scatter_rgb, ax=axes[1, 1], orientation='vertical', label='RGB')

# レイアウトを調整
plt.tight_layout()

# 画像を表示
plt.show()



        '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0206_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0207_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0208_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0209_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0210_filtered.csv'
"""