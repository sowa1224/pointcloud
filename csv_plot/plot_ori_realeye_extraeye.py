import sys
import pandas as pd
from my_functions import *
import os
import matplotlib.pyplot as plt
# 他の必要なモジュールのimportも行ってください

base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/'
real_eye_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_eye_data/real_eye_data/'
extracted_eye_path='/home/sowa/prog/pointcloud/csv/potato_1118/peel_eye_data/extracted_eye_data/'

# 処理対象の数字の範囲を指定
start_number = 2
end_number = 2

# 結果を保存するためのリスト
results = []

# ファイル処理の繰り返し
for number in range(start_number, end_number + 1, 1):
    # ファイル名の構築
    base_filename = f'potato_peel_1118_010{number}_filtered.csv'
    real_filename = f'peel_1118_010{number}.csv'
    eye_h_filename = f'eye_peel_1118_010{number}_h.csv'
    eye_s_filename = f'eye_peel_1118_010{number}_s.csv'
    eye_v_filename = f'eye_peel_1118_010{number}_v.csv'
    eye_hs_filename = f'eye_peel_1118_010{number}_hs.csv'
    eye_hv_filename = f'eye_peel_1118_010{number}_hv.csv'
    eye_sv_filename = f'eye_peel_1118_010{number}_sv.csv'
    eye_hsv_filename = f'eye_peel_1118_010{number}_hsv.csv'

    # ファイルの絶対パスを構築
    csv_file_path_base = base_path + base_filename
    csv_file_path_real = real_eye_path + real_filename
    csv_file_path_eye_h = extracted_eye_path + eye_h_filename
    csv_file_path_eye_s = extracted_eye_path + eye_s_filename
    csv_file_path_eye_v = extracted_eye_path + eye_v_filename
    csv_file_path_eye_hs = extracted_eye_path + eye_hs_filename
    csv_file_path_eye_hv = extracted_eye_path + eye_hv_filename
    csv_file_path_eye_sv = extracted_eye_path + eye_sv_filename
    csv_file_path_eye_hsv = extracted_eye_path + eye_hsv_filename
  
    
    
       # ファイルの読み込み
    base_data = pd.read_csv(csv_file_path_base, header=None)
    base_data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
    print(base_data)

    real_data = pd.read_csv(csv_file_path_real, header=None,skiprows=1)
    real_data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_h = pd.read_csv(csv_file_path_eye_h, header=None,skiprows=1)
    data_h.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_s = pd.read_csv(csv_file_path_eye_s, header=None,skiprows=1)
    data_s.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_v = pd.read_csv(csv_file_path_eye_v, header=None,skiprows=1)
    data_v.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_hs = pd.read_csv(csv_file_path_eye_hs, header=None,skiprows=1)
    data_hs.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_hv = pd.read_csv(csv_file_path_eye_hv, header=None,skiprows=1)
    data_hv.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_sv = pd.read_csv(csv_file_path_eye_sv, header=None,skiprows=1)
    data_sv.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

    data_hsv = pd.read_csv(csv_file_path_eye_hsv, header=None,skiprows=1)
    data_hsv.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
