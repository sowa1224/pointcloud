import open3d as o3d
import numpy as np
import plyfile
import matplotlib.pyplot as plt
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# CSVファイルを読み込む
data = np.loadtxt('/home/sowa/prog/pointcloud/csv/carrot.csv', delimiter=',')

# XYZ座標データを取得
#もとのcsvファイルは1行目はヘッダー（csvファイル内のデータの内容を表示する）のため
#色や法線も同様に、2行目から読み込む
points = data[:, 0:3]
print("すべてのポイントの3次元情報を表示")
print(points.shape)


# 色情報を取得
colors = data[:, 3:6]
print("すべてのポイントの色情報を表示")
print(colors)
print(colors.shape)


#法線ベクトルを取得
vec = data[:, 6:]
print("すべてのポイントの法線部クトル情報を表示")
print(vec.shape)

#もとのデータの大きさを表示
print("もとのデータの大きさを表示")
print(data.shape)


def rgb_to_hsv(rgb):
    # OpenCVのBGR形式に変換

    # BGRからHSVに変換
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    return hsv


# CSVファイルのパス
csv_file_path = '/home/sowa/prog/pointcloud/csv/carrot.csv'

# CSVファイルを読み込み、RGBデータを抽出
point_cloud = []

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        rgb_values = list(map(float, row[3:6]))  # RGBデータは4列から6列に格納されていると仮定
        rgb = np.array(rgb_values) * 255  # 0-1の範囲から0-255の範囲に変換
        point_cloud.append(rgb.astype(int))

# RGBデータをNumPy配列に変換
point_cloud = np.array(point_cloud, dtype=np.uint8)
print("0~1に変換後はこうなら",point_cloud)
print(point_cloud.shape)
print(point_cloud.dtype)

# ピクセルごとに変換するために配列をリシェイプ
point_cloud = point_cloud.reshape(-1, 1, 3)

# RGBからHSVに変換
hsv_data = rgb_to_hsv(point_cloud)
print("HSVデータ:\n", hsv_data)
print(hsv_data.shape)



# 色相の範囲を指定
min_hue = 15  # 最小色相（赤）
max_hue = 20  # 最大色相（オレンジ）

# 色相が指定範囲内のインデックスを取得
indices = np.where((hsv_data[:,0, 0] >= min_hue) & (hsv_data[:,0, 0] <= max_hue))[0]

# 色相が指定範囲内のデータを抽出
matching_hsv_data = hsv_data[indices]

# 結果の表示
print("Matching HSV data:")
print(matching_hsv_data,matching_hsv_data.shape)

print(matching_hsv_data[:,0,0])

selected_df = matching_hsv_data
import open3d as o3d
import numpy as np
import plyfile
import matplotlib.pyplot as plt
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# CSVファイルを読み込む
data = np.loadtxt('/home/sowa/prog/pointcloud/csv/carrot.csv', delimiter=',')

# XYZ座標データを取得
#もとのcsvファイルは1行目はヘッダー（csvファイル内のデータの内容を表示する）のため
#色や法線も同様に、2行目から読み込む
points = data[:, 0:3]
print("すべてのポイントの3次元情報を表示")
print(points.shape)


# 色情報を取得
colors = data[:, 3:6]
print("すべてのポイントの色情報を表示")
print(colors)
print(colors.shape)


#法線ベクトルを取得
vec = data[:, 6:]
print("すべてのポイントの法線部クトル情報を表示")
print(vec.shape)

#もとのデータの大きさを表示
print("もとのデータの大きさを表示")
print(data.shape)


def rgb_to_hsv(rgb):
    # OpenCVのBGR形式に変換

    # BGRからHSVに変換
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    return hsv


# CSVファイルのパス
csv_file_path = '/home/sowa/prog/pointcloud/csv/carrot.csv'

# CSVファイルを読み込み、RGBデータを抽出
point_cloud = []

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        rgb_values = list(map(float, row[3:6]))  # RGBデータは4列から6列に格納されていると仮定
        rgb = np.array(rgb_values) * 255  # 0-1の範囲から0-255の範囲に変換
        point_cloud.append(rgb.astype(int))

# RGBデータをNumPy配列に変換
point_cloud = np.array(point_cloud, dtype=np.uint8)
print("0~1に変換後はこうなら",point_cloud)
print(point_cloud.shape)
print(point_cloud.dtype)

# ピクセルごとに変換するために配列をリシェイプ
point_cloud = point_cloud.reshape(-1, 1, 3)

# RGBからHSVに変換
hsv_data = rgb_to_hsv(point_cloud)
print("HSVデータ:\n", hsv_data)
print(hsv_data.shape)



# 色相の範囲を指定
min_hue = 15  # 最小色相（赤）
max_hue = 20  # 最大色相（オレンジ）

# 色相が指定範囲内のインデックスを取得
indices = np.where((hsv_data[:,0, 0] >= min_hue) & (hsv_data[:,0, 0] <= max_hue))[0]

# 色相が指定範囲内のデータを抽出
matching_hsv_data = hsv_data[indices]

# 結果の表示
print("Matching HSV data:")
print(matching_hsv_data,matching_hsv_data.shape)

print(matching_hsv_data[:,0,0])

# 選択されたデータを使用してデータフレームを作成
selected_df = pd.DataFrame(matching_hsv_data[:, 0, :])
selected_df.columns = ['H', 'S', 'V']

# CSVファイルに保存
edited_csv_file = "/home/sowa/prog/pointcloud/csv/carrot_extratc.csv"
selected_df.to_csv(edited_csv_file, index=False)

""""

#HSVを保存
# HSVデータをPandas DataFrameに変換
df = pd.DataFrame(hsv_data, columns=["H", "S", "V"])

# CSVファイルとして保存
df.to_csv( '/home/sowa/prog/pointcloud/hsv/carrot_1.csv', index=False)
#次に法線ベクトルが001以外の点群を抽出する
#更に深度をxメートル以内の点群を出す（壁以外の点群を出す）
print("深度範囲")
print(data[:,3])
depth_input = int(input("深度の大きさは？"))
selected = data[(data[:, 3] <= depth_input) & (np.any(data[:,6:] != [0, 0, 1], axis=1))]
#selected = data[np.where((points[:,3] <= depth_input) & np.any(vec != [0, 0, 1], axis=1))[0]]
#selected = data[np.all(vec!=[0,0,1],axis=1)]


#抽出用の配列
selected_array = np.array([])

if selected.size >0:
    selected_array = selected

print("抽出した物体の点群の座標情報を表示")
print(selected_array.shape)
print(selected_array)

#次に抽出した配列を保存及び表示する
selected_points = selected_array[:,0:3]
print(selected_points)
selected_colors = selected_array[:,3:6]
selected_vec = selected_array[:,6:]
df = pd.DataFrame({
    'X': selected_points[:, 0],  # X座標の列
    'Y': selected_points[:, 1],  # Y座標の列
    'Z': selected_points[:, 2],  # Z座標の列
    'R': selected_colors [:, 0],  # 赤色成分の列
    'G': selected_colors [:, 1],  # 緑色成分の列
    'B': selected_colors [:, 2],   # 青色成分の列
    'nX':selected_vec [:,0],
    'nY':selected_vec [:,1],
    'nZ':selected_vec [:,2]
    # その他の列名とデータがあれば適宜追加する
})
print(selected_points.shape,selected_colors.shape,selected_vec.shape)
# CSVファイルとして保存
df.to_csv('extracted_point_cloud.csv', index=False)

#３次元で表示
# CSVファイルの読み込み
df = pd.read_csv('extracted_point_cloud.csv')


# XYZ座標データを取得
points = df[['X', 'Y', 'Z']].values

# Open3DのPointCloudデータに変換
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
print("深度は"+str(depth_input ))


#抽出前後のcsvファイルの行列の大きさを表示
def get_csv_size(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        num_rows = len(data)  # 行数
        num_columns = len(data[0])  # 列数（1行目のデータフィールド数）

    return num_rows, num_columns

csv_path1 = 'point_cloud_test.csv'
csv_path2 = 'pointcloud_extracted.csv'
rows1, cols1 = get_csv_size(csv_path1)
rows2, cols2 = get_csv_size(csv_path2)

print(f"CSVテストファイルの行数: {rows1}")
print(f"CSVテストファイルの列数: {cols1}")

print(f"CSV抽出ファイルの行数: {rows2}")
print(f"CSV抽出ファイルの列数: {cols2}")




#データを結合
selected_data = np.concatenate((selected_points,selected_colors,selected_vec),axis=1)
print(selected_data.shape)
print(selected_data)

# CSVファイルにデータを保存
np.savetxt('pointcloud_extracted.csv', selected_data, delimiter=',')


def get_csv_size(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        num_rows = len(data)
        if num_rows > 0:
            num_cols = len(data[0])
        else:
            num_cols = 0
    return num_rows, num_cols

# CSVファイルのパスを指定して、ファイルの大きさを取得する
csv_path1 = 'point_cloud_test.csv'
csv_path2 = 'pointcloud_extracted.csv'
rows1, cols1 = get_csv_size(csv_path1)
rows2, cols2 = get_csv_size(csv_path2)

print(f"CSVテストファイルの行数: {rows1}")
print(f"CSVテストファイルの列数: {cols1}")

print(f"CSV抽出ファイルの行数: {rows2}")
print(f"CSV抽出ファイルの列数: {cols2}")

#CSVファイルをPLYファイルに変換、保存、表示
# CSVファイルのパスと読み込む
df = pd.read_csv(csv_path1)
"""