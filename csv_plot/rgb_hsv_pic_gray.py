import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import rgb_to_hsv

file_paths = [
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0101_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0102_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0103_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0104_filtered.csv',
    '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0105_filtered.csv',
]

fig, axes = plt.subplots(5, 4, figsize=(15, 20))

for idx, path in enumerate(file_paths):
    data = pd.read_csv(path, header=None)
    data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    coordinates = data[['X', 'Y', 'Z']].values
    rgb_data = data[['R', 'G', 'B']].values / 255.0
    hsv_data = data[['H', 'S', 'V']].values

    hue_data = hsv_data[:, 0] / 255  # Hueを0~1にスケーリング（0~360度から）
    saturation_data = hsv_data[:, 1]   # Saturationを0~1にスケーリング
    value_data = hsv_data[:, 2]  # Valueを0~1にスケーリング

    num_points = data.shape[0]

    row = idx
    col = 0

    scatter_hue = axes[row, col].scatter(coordinates[:, 0], coordinates[:, 1], c=hue_data, cmap='gray', s=5)
    axes[row, col].set_xlabel('X')
    axes[row, col].set_ylabel('Y')
    cbar_hue = fig.colorbar(scatter_hue, ax=axes[row, col], orientation='vertical', label='Hue')
    cbar_hue.set_clim(0, 1)

    scatter_saturation = axes[row, col + 1].scatter(coordinates[:, 0], coordinates[:, 1], c=saturation_data, cmap='gray', s=5)
    axes[row, col + 1].set_xlabel('X')
    axes[row, col + 1].set_ylabel('Y')
    cbar_saturation = fig.colorbar(scatter_saturation, ax=axes[row, col + 1], orientation='vertical', label='Saturation')
    cbar_saturation.set_clim(0, 255)

    scatter_value = axes[row, col + 2].scatter(coordinates[:, 0], coordinates[:, 1], c=value_data, cmap='gray', s=5)
    axes[row, col + 2].set_xlabel('X')
    axes[row, col + 2].set_ylabel('Y')
    cbar_value = fig.colorbar(scatter_value, ax=axes[row, col + 2], orientation='vertical', label='Value')
    cbar_value.set_clim(0, 255)

    scatter_rgb = axes[row, col + 3].scatter(coordinates[:, 0], coordinates[:, 1], c=rgb_data, s=5)
    axes[row, col + 3].set_xlabel('X')
    axes[row, col + 3].set_ylabel('Y')

plt.tight_layout()
plt.show()
