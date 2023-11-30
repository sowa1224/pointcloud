import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import os
my_functions_path = "/home/sowa/prog/pointcloud/pointcloud_program/"
sys.path.append(my_functions_path)
from my_functions import *


# 抽出したデータを新しいCSVファイルに保存
original_path = "/home/sowa/prog/pointcloud/csv/potato_1118/peel_filtered/"
eye_path = "/home/sowa/prog/pointcloud/csv/potato_1118/peel_eye_data/real_eye_data/"

# 抽出する彩度の範囲
min_saturation, max_saturation = 100,140
saturation_range = (min_saturation,max_saturation)

# 抽出する明度の範囲
min_value, max_value = 90,160
value_range = (min_value,max_value)

# 処理対象の数字の範囲を指定
start_number = 1
end_number = 9

# 結果を保存するためのリスト
results = []

#ここはSの範囲内の
for number in range(start_number, end_number + 1, 1):
    # ファイル名の構築
    filtered_csv_filename = f'potato_peel_1118_020{number}_filtered.csv'
    eye_data_filename = f'peel_1118_020{number}.csv'

    # ファイルの絶対パスを構築
    csv_file_path = original_path + filtered_csv_filename
    csv_file_path_eye = eye_path + eye_data_filename

    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_eye):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 彩度に基づいて抽出
        extracted_data = extract_data_by_saturation(data, saturation_range)

        # 対応するunpeelファイルの読み込み
        eye_data = pd.read_csv(csv_file_path_eye, header=None,skiprows=1)
        eye_data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # サブプロットを作成し、1行2列の配置を行う
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 元のデータを散布図としてプロット（カラー表示）
        axs[0].scatter(data['X'], data['Y'], c=data[['R', 'G', 'B']].values / 255, label='Original Points')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')


        # 抽出した点を散布図としてプロット（白黒表示）
        axs[1].scatter(data['X'], data['Y'], color='black')
        axs[1].scatter(extracted_data['X'], extracted_data['Y'], color='white')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')


        # グラフを表示
        plt.tight_layout()
        plt.show()



        # 元のデータを散布図としてプロット（黒色で表示）
        #plt.scatter(data['X'], data['Y'], color='black', label='Original Points')

        # 抽出した点を散布図としてプロット（白色で表示）
        #plt.scatter(extracted_data['X'], extracted_data['Y'], color='white', label='Extracted Points')

        # グラフの装飾
        #plt.title('Scatter Plot of Original and Extracted Points')
        #plt.xlabel('X-axis')
        #plt.ylabel('Y-axis')
        #plt.legend()
        #plt.show()


        #print(data_unpeel)

        # Convert extracted_data to a DataFrame for easy comparison02
        extracted_data_df = pd.DataFrame(extracted_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        #extracted_data_df.to_csv(csv_file_path_eye,index=False)
        #extracted_data_df.to_csv('/home/sowa/prog/pointcloud/csv/potato_1118/extracted_data.csv', index=False)

        # データをマージして重複する行を抽出
        merged_data = pd.merge(extracted_data_df, eye_data, on=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'], how='inner')

        overlap_count = len(merged_data)
        #print(f"重なる点の数: {overlap_count}")

        # extracted_dataの合計点数を出力
        total_points_extracted_data = len(extracted_data_df)
        #print(f"extracted_dataの合計点数 ({filtered_csv_filename}): {total_points_extracted_data}")

        #実際の芽の点の数
        real_eye_points_count = len(eye_data)

        print(f"Sの範囲内：実際の点の割合は{overlap_count/real_eye_points_count}")




    else:
        print(f"ファイルが存在しないか、どちらか一方が存在しません ({filtered_csv_filename} または {csv_file_path_eye})")


#ここはVの範囲内
    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_eye):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 明度に基づいて抽出
        extracted_data = extract_data_by_value(data, value_range)

        # サブプロットを作成し、1行2列の配置を行う
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 元のデータを散布図としてプロット（カラー表示）
        axs[0].scatter(data['X'], data['Y'], c=data[['R', 'G', 'B']].values / 255, label='Original Points')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')


        # 抽出した点を散布図としてプロット（白黒表示）
        axs[1].scatter(data['X'], data['Y'], color='black')
        axs[1].scatter(extracted_data['X'], extracted_data['Y'], color='white')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')


        # グラフを表示
        plt.tight_layout()
        plt.show()


        # 対応するunpeelファイルの読み込み
        eye_data = pd.read_csv(csv_file_path_eye, header=None,skiprows=1)
        eye_data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        extracted_data_df = pd.DataFrame(extracted_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        # データをマージして重複する行を抽出
        merged_data = pd.merge(extracted_data_df, eye_data, on=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'], how='inner')
        overlap_count = len(merged_data)
        total_points_extracted_data = len(extracted_data_df)
        
        #実際の芽の点の数
        real_eye_points_count = len(eye_data)

        print(f"Vの範囲内：実際の点の割合は{overlap_count/real_eye_points_count}")




    else:
        print(f"ファイルが存在しないか、どちらか一方が存在しません ({filtered_csv_filename} または {csv_file_path_eye})")


#ここはSの範囲の点とVの範囲の点を2つ合わせたに持つ場合
    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_eye):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 明度に基づいて抽出
        extracted_data_s = extract_data_by_saturation(data,value_range)
        extracted_data_v = extract_data_by_value(data, value_range)

                # extracted_data_sとextracted_data_vを結合
        combined_data = pd.concat([extracted_data_s, extracted_data_v])

        # 重複する行を除外
        unique_data = combined_data.drop_duplicates()
        # 元のデータを散布図としてプロット（黒色で表示）
        # サブプロットを作成し、1行2列の配置を行う
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 元のデータを散布図としてプロット（カラー表示）
        axs[0].scatter(data['X'], data['Y'], c=data[['R', 'G', 'B']].values / 255, label='Original Points')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')


        # 抽出した点を散布図としてプロット（白黒表示）
        axs[1].scatter(data['X'], data['Y'], color='black')
        axs[1].scatter(unique_data['X'], unique_data['Y'], color='white')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')
   

        # グラフを表示
        plt.tight_layout()
        plt.show()

        # 対応するeyeファイルの読み込み
        eye_data = pd.read_csv(csv_file_path_eye, header=None,skiprows=1)
        eye_data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        extracted_data_df = pd.DataFrame(unique_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        # データをマージして重複する行を抽出
        merged_data = pd.merge(extracted_data_df, eye_data, on=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'], how='inner')
        overlap_count = len(merged_data)
        total_points_extracted_data = len(extracted_data_df)
        
        #実際の芽の点の数
        real_eye_points_count = len(eye_data)

        print(f"SとVの範囲内：実際の点の割合は{overlap_count/real_eye_points_count}")




    else:
        print(f"ファイルが存在しないか、どちらか一方が存在しません ({filtered_csv_filename} または {csv_file_path_eye})")

#ここはSとVの範囲の2条件を満たした場合
    if os.path.exists(csv_file_path) and os.path.exists(csv_file_path_eye):

        # ファイルの読み込み
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 明度に基づいて抽出
        extracted_data_s = extract_data_by_saturation(data,value_range)
        extracted_data_sv = extract_data_by_value(extracted_data_s, value_range)

        # 元のデータを散布図としてプロット（黒色で表示）
        # サブプロットを作成し、1行2列の配置を行う
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 元のデータを散布図としてプロット（カラー表示）
        axs[0].scatter(data['X'], data['Y'], c=data[['R', 'G', 'B']].values / 255, label='Original Points')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')


        # 抽出した点を散布図としてプロット（白黒表示）
        axs[1].scatter(data['X'], data['Y'], color='black')
        axs[1].scatter(extracted_data_sv['X'], extracted_data_sv['Y'], color='white')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')
   

        # グラフを表示
        plt.tight_layout()
        plt.show()

        # 対応するeyeファイルの読み込み
        eye_data = pd.read_csv(csv_file_path_eye, header=None,skiprows=1)
        eye_data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']
        extracted_data_df = pd.DataFrame(extracted_data_sv, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'])
        # データをマージして重複する行を抽出
        merged_data = pd.merge(extracted_data_df, eye_data, on=['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V'], how='inner')
        overlap_count = len(merged_data)
        total_points_extracted_data = len(extracted_data_df)
        
        #実際の芽の点の数
        real_eye_points_count = len(eye_data)

        print(f"S＆Vの範囲内：実際の点の割合は{overlap_count/real_eye_points_count}")




    else:
        print(f"ファイルが存在しないか、どちらか一方が存在しません ({filtered_csv_filename} または {csv_file_path_eye})")





