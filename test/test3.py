import cv2
import pandas as pd
import numpy as np

# CSVファイルからデータを読み込む
file_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_0103_filtered.csv'  # ファイルパスを変更してください
data = pd.read_csv(file_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

# 各点のX、Y、Zの値を抽出する
x_values = data['X'].astype(float)
y_values = data['Y'].astype(float)
z_values = data['Z'].astype(float)

# 座標の最小値を求める
x_min = x_values.min()
y_min = y_values.min()

# 座標を画像の位置にマッピングする
data['Mapped_X'] = (x_values - x_min).astype(int)
data['Mapped_Y'] = (y_values - y_min).astype(int)

# 画像のサイズを決定する
width = data['Mapped_X'].max() + 1
height = data['Mapped_Y'].max() + 1

# 新しい画像を作成
image = np.zeros((height, width, 3), dtype=np.uint8)

# 点の数だけループして画像を作成する
for i in range(len(data)):
    x = data['Mapped_X'][i]
    y = data['Mapped_Y'][i]
    z = int(z_values[i])
    r = int(data['R'][i])
    g = int(data['G'][i])
    b = int(data['B'][i])

    if 0 <= y < height and 0 <= x < width:
        image[y, x] = [b, g, r]  # OpenCVでは(y, x)の順番で座標を指定することに注意

# 画像を表示する（任意）
cv2.imshow('Point Cloud Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像をファイルに保存する（任意）
cv2.imwrite('/home/sowa/prog/pointcloud/RGB,DEPTH/RGB/point_cloud_image.png', image)
