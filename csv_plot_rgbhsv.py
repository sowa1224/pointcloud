import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む（ここではパスを'csv_file_path'として置き換えてください）
csv_file_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_eye_data/unpeel_1118_01combined.csv'
data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
# RGBとHSVのデータを取得
r_data = data['R']
g_data = data['G']
b_data = data['B']
h_data = data['H']
s_data = data['S']
v_data = data['V']

# グラフを作成
fig, axes = plt.subplots(2, 3, figsize=(12, 12))
axes = axes.flatten()

colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

for i, (data, label, color) in enumerate(zip([r_data, g_data, b_data, h_data, s_data, v_data], ['R', 'G', 'B', 'H', 'S', 'V'], colors)):
    axes[i].plot(data, marker='o', linestyle='', color=color)
    axes[i].set_ylabel(label)
    axes[i].set_xlabel('Point Number')
    axes[i].set_title(f'{label}')

plt.tight_layout()
plt.show()