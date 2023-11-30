import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program/"

# sys.path に my_functions.py のディレクトリのパスを追加します。
sys.path.append(my_functions_path)
from my_functions import *

peel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/potato_peel_1118_0101_filtered.csv'
unpeel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/'
fig_directory = '/home/sowa/prog/pointcloud/picture/1118_eye_extract_判断材料/s_all/'

min_saturation, max_saturation = 70, 240
min_value, max_value = 70, 240
saturation_range=(min_saturation,min_saturation+10)

data = pd.read_csv(peel_base_path, header=None)
data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
coordinates = data[['X', 'Y']].values * 10000000  # 座標を10,000,000倍に
rgb_data = data[['R', 'G', 'B']].values / 255.0

extracted_data = extract_data_by_saturation(data, saturation_range)

# 仮の座標データ（黒い点）
black_coordinates = coordinates
# 仮の座標データ（白い点）
white_coordinates = extracted_data[['X', 'Y']].values * 10000000  # 座標を10,000,000倍に

# 画像サイズの設定
image_width, image_height = 200, 200
image = Image.new("RGB", (image_width, image_height), (255, 255, 255))  # 白色の画像を作成

# 座標を元に画像に点を描画する関数
def draw_points_on_image(coordinates, color):
    draw = ImageDraw.Draw(image)
    for coord in coordinates:
        x, y = coord
        draw.point((x, y), fill=color)

# 黒い点を描画（黒色の点）
draw_points_on_image(black_coordinates, (0, 0, 0))

# 白い点を描画（白色の点）
draw_points_on_image(white_coordinates, (255, 255, 255))

# 画像の表示
image.show()
# 画像の保存
image.save("/home/sowa/prog/pointcloud/picture/black_white_points_image.png")
