import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program/"

# sys.path に my_functions.py のディレクトリのパスを追加します。
sys.path.append(my_functions_path)
from my_functions import *

peel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/'
unpeel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/'
fig_directory_s = '/home/sowa/prog/pointcloud/picture/1118_eye_extract_判断材料/s_all/'
fig_directory_v = '/home/sowa/prog/pointcloud/picture/1118_eye_extract_判断材料/v_all/'
# 抽出する彩度の範囲
min_saturation, max_saturation = 70, 200

# 抽出する明度の範囲
min_value, max_value = 70, 200
step = 10  # ステップサイズを設定

# サブプロット用の図のサイズ
num_rows = 3  # 行数
num_cols = 6  # 列数

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))


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
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))    
        for i, saturation in enumerate(range(min_saturation, max_saturation, step)):
            row = i // num_cols
            col = i % num_cols

            saturation_range = (saturation, saturation + step)
            extracted_data = extract_data_by_saturation(data, saturation_range)

            axs[row, col].scatter(data['X'], data['Y'], color='black')
            axs[row, col].scatter(extracted_data['X'], extracted_data['Y'], color='white')
            #axs[row, col].scatter(extracted_data['X'], extracted_data['Y'], color='white',edgecolors='black')
            axs[row, col].set_title(f'V Range:{saturation}-{saturation + step}')
            axs[row, col].set_xlabel('X')
            axs[row, col].set_ylabel('Y')

        axs[2, 5].scatter(coordinates[:, 0], coordinates[:, 1], c=rgb_data, s=5)
        axs[2, 5].set_title('Original')
        axs[2, 5].set_xlabel('X')
        axs[2, 5].set_ylabel('Y')
        plt.tight_layout()
        # 画像の保存
        output_filename = f'peel_010{number}_s_70~200.png'  # 画像の名前
        output_path = os.path.join(fig_directory_s, output_filename)
        plt.savefig(output_path)  # 画像を保存
        plt.close()  # プロットを閉じる


    if os.path.exists(csv_file_path_peel) and os.path.exists(csv_file_path_unpeel):
        data_2 = pd.read_csv(csv_file_path_peel, header=None)
        data_2.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        coordinates = data_2[['X', 'Y']].values
        rgb_data = data_2[['R', 'G', 'B']].values / 255.0
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))    
        for i, value in enumerate(range(min_value, max_value, step)):
            row = i // num_cols
            col = i % num_cols

            value_range = (value, value + step)
            extracted_data = extract_data_by_value(data_2, value_range)

            axs[row, col].scatter(data['X'], data['Y'], color='black')
            axs[row, col].scatter(extracted_data['X'], extracted_data['Y'], color='white')

            axs[row, col].set_title(f'V Range:{value}-{value + step}')
            axs[row, col].set_xlabel('X')
            axs[row, col].set_ylabel('Y')

        axs[2, 5].scatter(coordinates[:, 0], coordinates[:, 1], c=rgb_data, s=5)
        axs[2, 5].set_title('Original')
        axs[2, 5].set_xlabel('X')
        axs[2, 5].set_ylabel('Y')
        plt.tight_layout()
        # 画像の保存
        output_filename = f'peel_010{number}_v_70~200.png'  # 画像の名前
        output_path = os.path.join(fig_directory_v, output_filename)
        plt.savefig(output_path)  # 画像を保存
        plt.close()  # プロットを閉じる





