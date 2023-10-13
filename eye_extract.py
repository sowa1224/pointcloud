import numpy as np
import pandas as pd
import cv2
import open3d as o3d

# CSVファイルを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_filtered_1011.csv"
df = pd.read_csv(csv_file_path, header=None)
df.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'nX', 'nY', 'nZ']

# RGBデータをHSVデータに変換する関数
def rgb_to_hsv(row):
    rgb = np.array(row[['R', 'G', 'B']])
    rgb = rgb * 255
    hsv = cv2.cvtColor(rgb.astype('uint8').reshape(1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv[0, 0]

# Apply the function to each row and extract the 'V' component
#df['HSV'] = df.apply(rgb_to_hsv, axis=1)
data = df
hsv_data = data[['R', 'G', 'B']].apply(rgb_to_hsv, axis=1)
#df['V'] = df['HSV'].apply(lambda x: x[2])

# Display the 'V' values
print(hsv_data)