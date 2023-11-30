import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from PIL import Image, ImageDraw

my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program/"

# sys.path に my_functions.py のディレクトリのパスを追加します。
sys.path.append(my_functions_path)
from my_functions import *

peel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/'
unpeel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/'
fig_directory = '/home/sowa/prog/pointcloud/picture/1118_eye_extract_判断材料/s_all/'
# 抽出する彩度の範囲
min_saturation, max_saturation = 70, 240

# 抽出する明度の範囲
min_value, max_value = 70, 240
step = 10  # ステップサイズを設定


# ファイル処理の繰り返し
for number in range(1,10, 1):
    # ファイル名の構築
    peel_csv_filename = f'potato_peel_1118_010{number}_filtered.csv'
    unpeel_csv_filename = f'potato_un_1118_010{number}_filtered.csv'

    # ファイルの絶対パスを構築
    csv_file_path_peel = peel_base_path+ peel_csv_filename
    csv_file_path_unpeel = unpeel_base_path + unpeel_csv_filename

    if os.path.exists(csv_file_path_peel) and os.path.exists(csv_file_path_unpeel):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path_peel, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        coordinates = data[['X', 'Y']].values
        rgb_data = data[['R', 'G', 'B']].values / 255.0

        for i, saturation in enumerate(range(min_saturation, max_saturation, step)):
            fig, ax = plt.subplots(figsize=(8, 6))  # サブプロットを作成せずに図を直接作成

            saturation_range = (saturation, saturation + step)
            extracted_data = extract_data_by_saturation(data, saturation_range)

            plt.scatter(data['X'], data['Y'], color='black')
            plt.scatter(extracted_data['X'], extracted_data['Y'], color='white')

            plt.title(f'S Range:{saturation}-{saturation + step}')
            plt.xlabel('X')
            plt.ylabel('Y')

            output_filename = f'peel_010{number}_s_{saturation}-{saturation+step}.png'
            output_path = os.path.join(fig_directory, output_filename)
            plt.savefig(output_path)  # 画像を保存
            plt.close()  # 図を閉じる