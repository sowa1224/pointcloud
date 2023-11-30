import sys
import pandas as pd
from my_functions import *
import os

# 他の必要なモジュールのimportも行ってください

base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_filtered/'
unpeel_base_path  = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_eye_data/real_eye_data/'
eye_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_eye_data/extracted_eye_data/'
# 色相に基づく抽出の設定
min_hue, max_hue = 18, 45
hue_range = (min_hue, max_hue)

# 彩度に基づく抽出の設定
max_saturation = 180
min_saturation = 100
saturation_range = (min_saturation, max_saturation)

# 抽出する明度の範囲
min_value, max_value = 60, 115
value_range = (min_value, max_value)


# 処理対象の数字の範囲を指定
start_number = 1
end_number = 9

# 結果を保存するためのリスト
results = []

# ファイル処理の繰り返し
for number in range(start_number, end_number + 1, 1):
    # ファイル名の構築
    filtered_csv_filename = f'potato_un_1118_020{number}_filtered.csv'
    unpeel_csv_filename = f'un_1118_020{number}.csv'
    eye_data_save_filename_h = f'eye_unpeel_1118_020{number}_h.csv'
    eye_data_save_filename_s = f'eye_unpeel_1118_020{number}_s.csv'
    eye_data_save_filename_v = f'eye_unpeel_1118_020{number}_v.csv'
    eye_data_save_filename_hs = f'eye_unpeel_1118_020{number}_hs.csv'
    eye_data_save_filename_hv = f'eye_unpeel_1118_020{number}_hv.csv'
    eye_data_save_filename_sv = f'eye_unpeel_1118_020{number}_sv.csv'
    eye_data_save_filename_hsv = f'eye_unpeel_1118_020{number}_hsv.csv'

    # ファイルの絶対パスを構築
    csv_file_path = base_path + filtered_csv_filename
    csv_file_path_unpeel = unpeel_base_path + unpeel_csv_filename
    csv_file_path_eye_h = eye_base_path + eye_data_save_filename_h
    csv_file_path_eye_s = eye_base_path + eye_data_save_filename_s
    csv_file_path_eye_v = eye_base_path + eye_data_save_filename_v
    csv_file_path_eye_hs = eye_base_path + eye_data_save_filename_hs
    csv_file_path_eye_hv = eye_base_path + eye_data_save_filename_hv
    csv_file_path_eye_sv = eye_base_path + eye_data_save_filename_sv    
    csv_file_path_eye_hsv = eye_base_path + eye_data_save_filename_hsv

    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_unpeel):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 色相に基づいて抽出
        extracted_data = extract_data_by_hue(data, hue_range)

        # 彩度に基づいて抽出
        extracted_data_1 = extract_data_by_saturation(data,saturation_range)

        # 明度に基づいてデータを抽出
        extracted_data_2 = extract_data_by_value(data,value_range)

        #色相と彩度3、色相と明度4、彩度と明度5、３つ同時6
        extracted_data_3 = extract_data_by_saturation(extracted_data,saturation_range)
        extracted_data_4 = extract_data_by_value(extracted_data,value_range)        
        extracted_data_5 = extract_data_by_value(extracted_data_1,value_range)
        extracted_data_6 = extract_data_by_value(extracted_data_3,value_range)              
        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_h = pd.DataFrame(extracted_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_h.to_csv(csv_file_path_eye_h,index=False)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_s = pd.DataFrame(extracted_data_1, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_s.to_csv(csv_file_path_eye_s,index=False)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_v = pd.DataFrame(extracted_data_2, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_v.to_csv(csv_file_path_eye_v,index=False)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_hs = pd.DataFrame(extracted_data_3, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_hs.to_csv(csv_file_path_eye_hs,index=False)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_hv = pd.DataFrame(extracted_data_4, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_hv.to_csv(csv_file_path_eye_hv,index=False)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_sv = pd.DataFrame(extracted_data_5, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_sv.to_csv(csv_file_path_eye_sv,index=False)


        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_hsv = pd.DataFrame(extracted_data_6, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_hsv.to_csv(csv_file_path_eye_hsv,index=False)