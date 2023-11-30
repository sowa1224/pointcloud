import sys
import pandas as pd
from my_functions import *
import os

# 他の必要なモジュールのimportも行ってください

base_path = '/home/sowa/prog/pointcloud/csv/potato_1113/peel_filtered/'
eye_path = '/home/sowa/prog/pointcloud/csv/potato_1113/peel_eye_data/'

# 彩度に基づく抽出の設定
max_saturation = 255
min_saturation = 130
saturantion_range = (min_saturation, max_saturation)

# 処理対象の数字の範囲を指定
start_number = 0
end_number = 6
# 結果を保存するためのリスト
results = []

# ファイル処理の繰り返し
for number in range(start_number, end_number + 1, 3):
    # ファイル名の構築
    filtered_csv_filename = f'potato_peel_1113_{number}_filtered.csv'
    eye_filename = f'peel_1113_{number}.csv'

    # ファイルの絶対パスを構築
    csv_file_path = base_path + filtered_csv_filename
    csv_file_path_eye = eye_path + eye_filename

    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_eye):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 彩度に基づいて抽出
        extracted_data = extract_data_by_saturation(data, saturantion_range)
        
        # 対応するunpeelファイルの読み込み
        data_eye = pd.read_csv(csv_file_path_eye, skiprows=1, header=None)
        data_eye.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 列の絞り込み
        columns_to_compare = ['X', 'Y', 'Z', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        extracted_data_df = pd.DataFrame(extracted_data, columns=columns_to_compare)
        data_eye_df = pd.DataFrame(data_eye, columns=columns_to_compare)

        # 重なるデータを抽出
        merged_data = pd.concat([data_eye_df, extracted_data_df], axis=1, join='inner')

        # 重なる点の数を出力
        overlap_count = len(merged_data)
        print(f"重なる点の数 ({filtered_csv_filename} と {eye_filename}): {overlap_count}")

        # extracted_dataの合計点数を出力
        total_points_extracted_data = len(extracted_data_df)
        print(f"extracted_dataの合計点数 ({filtered_csv_filename}): {total_points_extracted_data}")

        # 精度の計算
        if overlap_count >= total_points_extracted_data:
            accuracy = (total_points_extracted_data / overlap_count) * 100
            print(f"精度 {accuracy}")
        else:
            accuracy = (overlap_count / total_points_extracted_data) * 100
            print(f"精度 {accuracy}")

        # 結果をリストに追加
        results.append({
            'FileNumber': number,
            'OverlapCount': overlap_count,
            'TotalPointsExtracted': total_points_extracted_data,
            'Accuracy': accuracy
        })

    else:
        print(f"ファイルが存在しないか、どちらか一方が存在しません ({filtered_csv_filename} または {eye_filename})")

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# 結果をCSVファイルとして保存
results_df.to_csv('/home/sowa/prog/pointcloud/csv/potato_1118/result/peel_results_s.csv', index=False)
