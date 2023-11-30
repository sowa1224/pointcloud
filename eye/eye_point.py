import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import numpy as np

# CSVファイルからデータを読み込む
csv_file_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/potato_un_1118_0210_filtered.csv'

data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

points = data[['X', 'Y']].values
colors = data[['R', 'G', 'B']].values / 255

fig, ax = plt.subplots()
scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, marker='o')

# CSVファイル保存先ディレクトリ
output_csv_dir = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_eye_data/'

# クリックされた座標を保持するためのリスト
clicked_points = []

# mplcursorsを使用してクリックしたときに関数を呼び出す
cursor = mplcursors.cursor()

@cursor.connect("add")
def on_click(sel):
    index = sel.target.index
    x = data['X'][index]
    y = data['Y'][index]
    z = data['Z'][index]
    r = data['R'][index]
    g = data['G'][index]
    b = data['B'][index]
    nx = data['nX'][index]
    ny = data['nY'][index]
    nz = data['nZ'][index]
    h = data['H'][index]
    s = data['S'][index]
    v = data['V'][index]

    print(f'クリックした座標: ({x}, {y})')
    print(f'RGB値: ({r}, {g}, {b})')

    # クリックされた座標がすでにリストに存在しない場合にのみ追加
    if not any(p['X'] == x and p['Y'] == y for p in clicked_points):
        clicked_points.append({'X': x, 'Y': y, 'Z': z, 'R': r, 'G': g, 'B': b, 'nX': nx, 'nY': ny, 'nZ': nz, 'H': h, 'S': s, 'V': v})

        # CSVファイルに座標を保存
        output_csv_path = f'{output_csv_dir}un_1118_0210.csv'
        clicked_point_data = pd.DataFrame(clicked_points)
        clicked_point_data.to_csv(output_csv_path, index=False)
        print(f'座標を {output_csv_path} に保存しました。')

# カラーバーを追加
cbar = plt.colorbar(scatter, ax=ax, label='RGB Color')

# プロットを表示
plt.show()