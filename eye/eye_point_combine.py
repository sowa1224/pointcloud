import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
import os

# CSVファイルのあるディレクトリを指定
csv_dir = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_eye_data/'
# 保存先のCSVファイルパス
output_csv_path = '/home/sowa/prog/pointcloud/csv/potato_1118/unpeel_eye_data/unpeel_1118_02combined.csv'
# 空のDataFrameを作成
combined_data = pd.DataFrame()

# ディレクトリ内の全てのCSVファイルを処理
for filename in os.listdir(csv_dir):
    if filename.endswith('.csv') and filename.startswith('un_1118_02'):  # ファイル名の条件を指定
        file_path = os.path.join(csv_dir, filename)
        
        # CSVファイルを読み込み
        data = pd.read_csv(file_path, header=None)
        
        # 結合
        combined_data = pd.concat([combined_data, data])

combined_data.to_csv(output_csv_path, index=False, header=False)
print(f'結合されたデータを {output_csv_path} に保存しました。')



