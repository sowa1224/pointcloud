import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#ここからが本来の内容
csv_file_path ="/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_0201_filtered.csv"
data = pd.read_csv(csv_file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ','H','S','V']

# サンプルの点群データ (x, y, z, r, g, b)
# 点群データをnumpy配列として取得
points = data[['X', 'Y', 'Z']].values
colors=data[['R', 'G', 'B']].values/255
pointss = np.concatenate((points,colors), axis=1)

# 新しい3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:,0],points[:,1],points[:,2],c=colors,marker='o')

# 軸ラベル
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# グラフを表示
plt.show()