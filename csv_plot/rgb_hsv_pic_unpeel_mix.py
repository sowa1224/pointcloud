import pandas as pd
import matplotlib.pyplot as plt


# CSVファイルのパス
eye_file_path = '/home/sowa/prog/pointcloud/csv/potato_1113/unpeel_eye_data/unpeel_1113_0.csv'
original_file_path_2 = '/home/sowa/prog/pointcloud/csv/potato_1113/unpeel_filtered/potato_unpeel_1113_0_filtered.csv'

# CSVファイルの読み込み
data = pd.read_csv(eye_file_path, skiprows=1)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

data_2 = pd.read_csv(original_file_path_2, header=None)
data_2.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']



# 2行3列のサブプロット
fig, axes = plt.subplots(2, 3, figsize=(15, 10))


# データ2のプロット
axes[0, 0].scatter(range(len(data_2)), data_2['R'].values, marker='o', color='b', label='original')
axes[0, 1].scatter(range(len(data_2)), data_2['G'].values, marker='o', color='b', label='original')
axes[0, 2].scatter(range(len(data_2)), data_2['B'].values, marker='o', color='b', label='original')
axes[1, 0].scatter(range(len(data_2)), data_2['H'].values, marker='o', color='b', label='original')
axes[1, 1].scatter(range(len(data_2)), data_2['S'].values, marker='o', color='b', label='original')
axes[1, 2].scatter(range(len(data_2)), data_2['V'].values, marker='o', color='b', label='original')
# データ1のプロット
axes[0, 0].scatter(range(len(data)), data['R'].values, marker='o', color='r', label='eye')
axes[0, 1].scatter(range(len(data)), data['G'].values, marker='o', color='r', label='eye')
axes[0, 2].scatter(range(len(data)), data['B'].values, marker='o', color='r', label='eye')
axes[1, 0].scatter(range(len(data)), data['H'].values, marker='o', color='r', label='eye')
axes[1, 1].scatter(range(len(data)), data['S'].values, marker='o', color='r', label='eye')
axes[1, 2].scatter(range(len(data)), data['V'].values, marker='o', color='r', label='eye')



# グラフの装飾
axes[0, 0].set_title('R')
axes[0, 1].set_title('G')
axes[0, 2].set_title('B')
axes[1, 0].set_title('H')
axes[1, 1].set_title('S')
axes[1, 2].set_title('V')

for ax in axes.flatten():
    ax.grid(True)

plt.tight_layout()
plt.show()