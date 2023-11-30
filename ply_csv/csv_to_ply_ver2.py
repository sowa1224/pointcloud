import csv
import pandas as pd
from pyntcloud import PyntCloud

# CSVファイルからデータを読み込む
data = pd.read_csv("/home/sowa/prog/pointcloud/csv/potato_filtered_1017_peeling1_doublenoise.csv")
ply_file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_filtered_1017_peeling1_doublenoise.ply"

x = data["X"]
y = data["Y"]
z = data["Z"]
r = data["R"]
g = data["G"]
b = data["B"]
nx = data["nX"]
ny = data["nY"]
nz = data["nZ"]

# 点の座標と法線ベクトルを組み合わせてPyntCloudオブジェクトを作成
cloud = PyntCloud(pd.DataFrame({
    "x": x, "y": y, "z": z,
    "red": r, "green": g, "blue": b,
    "nx": nx, "ny": ny, "nz": nz
}))

# PLYファイルに保存
cloud.to_file(ply_file_path, as_text=True)