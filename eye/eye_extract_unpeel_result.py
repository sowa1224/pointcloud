import sys
import pandas as pd
from my_functions import *
import os

# 他の必要なモジュールのimportも行ってください

base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/'
unpeel_base_path = '/home/sowa/prog/pointcloud/csv/potato_1118/peel_eye_data/'

# 色相に基づく抽出の設定
min_hue, max_hue = 18, 45
hue_range = (min_hue, max_hue)

# 彩度に基づく抽出の設定
max_saturation = 180
min_saturation = 100
saturantion_range = (min_saturation, max_saturation)

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
    filtered_csv_filename = f'potato_peel_1118_010{number}_filtered.csv'
    unpeel_csv_filename = f'peel_1118_010{number}.csv'
    eye_data_save_filename = f'eye_1118_010{number}.csv'

    # ファイルの絶対パスを構築
    csv_file_path = base_path + filtered_csv_filename
    csv_file_path_unpeel = unpeel_base_path + unpeel_csv_filename
    csv_file_path_eye = unpeel_base_path + eye_data_save_filename

    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_unpeel):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 色相に基づいて抽出
        extracted_data = extract_data_by_hue(data, hue_range)

        # 彩度に基づいて抽出
        #extracted_data_2 = extract_data_by_saturation(extracted_data,saturantion_range)

        # 明度に基づいてデータを抽出
        extracted_data_2 = extract_data_by_value(extracted_data,value_range)

        # 対応するunpeelファイルの読み込み
        data_unpeel = pd.read_csv(csv_file_path_unpeel, header=None,skiprows=1)
        data_unpeel.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        #print(data_unpeel)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_df = pd.DataFrame(extracted_data_2, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        extracted_data_df.to_csv(csv_file_path_eye,index=False)

        #extracted_data_df.to_csv('/home/sowa/prog/pointcloud/csv/potato_1118/extracted_data.csv', index=False)

        # データをマージして重複する行を抽出
        merged_data = pd.merge(extracted_data_df, data_unpeel, on=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'], how='inner')

        overlap_count = len(merged_data)
        print(f"重なる点の数: {overlap_count}")
        # 重複する行の数をカウント
        overlap_count = len(merged_data)
        print(f"重なる点の数: {overlap_count}")

        # extracted_dataの合計点数を出力
        total_points_extracted_data = len(extracted_data_df)
        print(f"extracted_dataの合計点数 ({filtered_csv_filename}): {total_points_extracted_data}")

        # 精度の計算
        if total_points_extracted_data != 0:
            if overlap_count >= total_points_extracted_data:
                Accuracy = (total_points_extracted_data / overlap_count) * 100
            else: 
                Accuracy = (overlap_count / total_points_extracted_data) * 100

            # 結果をリストに追加
            results.append({
                'FileNumber': number,
                'OverlapCount': overlap_count,
                'TotalPointsExtracted': total_points_extracted_data,
                'Accuracy': Accuracy
            })
        else:
            results.append({
                'FileNumber': number,
                'OverlapCount': overlap_count,
                'TotalPointsExtracted': total_points_extracted_data,
                'Accuracy': 0
            })


    else:
        print(f"ファイルが存在しないか、どちらか一方が存在しません ({filtered_csv_filename} または {unpeel_csv_filename})")

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# 結果をCSVファイルとして保存
results_df.to_csv('/home/sowa/prog/pointcloud/csv/potato_1118/result/unpeel_results_01_hv.csv', index=False)
print(f"結果はpeel_results_01_hv.csvに保存した")
