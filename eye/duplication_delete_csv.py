import os
import pandas as pd

# CSVファイルが格納されているディレクトリ
csv_directory = '/home/sowa/prog/pointcloud/csv/potato_1113/unpeel_eye_data/'

# ディレクトリ内のCSVファイルを処理
for file_name in os.listdir(csv_directory):
    if file_name.endswith(".csv"):
        # CSVファイルのパス
        csv_file_path = os.path.join(csv_directory, file_name)

        # データを読み込む
        data = pd.read_csv(csv_file_path, header=None)
        data.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ', 'H', 'S', 'V']

        # 重複した行を削除 
        data_no_duplicates = data.drop_duplicates()

        # 元のファイルに上書き保存
        data_no_duplicates.to_csv(csv_file_path, index=False,header=None)

print("処理が完了しました。")
