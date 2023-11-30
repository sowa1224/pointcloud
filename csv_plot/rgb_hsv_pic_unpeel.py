import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパス
csv_file_path = '/home/sowa/prog/pointcloud/csv/potato_1113/unpeel_eye_data/unpeel_combined.csv'
# CSVファイルの読み込み
data = pd.read_csv(csv_file_path,skiprows=1)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

# 2行3列のサブプロット
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Rのプロット
axes[0, 0].scatter(range(len(data)), data['R'].values, marker='o', color='r')
axes[0, 0].set_title('Red vs Peel Point Number')
axes[0, 0].set_xlabel('Point Number')
axes[0, 0].set_ylabel('Red')
axes[0,0].set_yticks(range(0, 256, 5))
# Gのプロット
axes[0, 1].scatter(range(len(data)), data['G'].values, marker='o', color='g')
axes[0, 1].set_title('Green vs Peel Point Number')
axes[0, 1].set_xlabel('Point Number')
axes[0, 1].set_ylabel('Green')
axes[0,1].set_yticks(range(0, 256, 5))
# Bのプロット
axes[0, 2].scatter(range(len(data)), data['B'].values, marker='o', color='b')
axes[0, 2].set_title('Blue vs Peel Point Number')
axes[0, 2].set_xlabel('Point Number')
axes[0, 2].set_ylabel('Blue')
axes[0,2].set_yticks(range(0, 256, 5))
# Hのプロット
axes[1, 0].scatter(range(len(data)), data['H'].values, marker='o', color='c')
axes[1, 0].set_title('Hue vs Peel Point Number')
axes[1, 0].set_xlabel('Point Number')
axes[1, 0].set_ylabel('Hue')
axes[1,0].set_yticks(range(0, 256, 5))

# Sのプロット
axes[1, 1].scatter(range(len(data)), data['S'].values, marker='o', color='m')
axes[1, 1].set_title('Saturation vs Peel Point Number')
axes[1, 1].set_xlabel('Point Number')
axes[1, 1].set_ylabel('Saturation')
axes[1,1].set_yticks(range(0, 256, 5))

# Vのプロット
axes[1, 2].scatter(range(len(data)), data['V'].values, marker='o', color='y')
axes[1, 2].set_title('Value vs Peel Point Number')
axes[1, 2].set_xlabel('Point Number')
axes[1, 2].set_ylabel('Value')
axes[1,2].set_yticks(range(0, 256, 5))
plt.tight_layout()
plt.show()
