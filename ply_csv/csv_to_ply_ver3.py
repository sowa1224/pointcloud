import csv
import pandas as pd
from pyntcloud import PyntCloud

# CSVファイルから点群データを読み込む
csv_file = "/home/sowa/prog/pointcloud/csv/potato_filtered_1011.csv"
ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/csv_to_ply_test_1015_v3.ply"


# ヘッダー情報
header = """ply
format ascii 1.0
comment pointcloud saved from Realsense Viewer
element vertex {num_vertices}
property float32 x
property float32 y
property float32 z
property float32 nx
property float32 ny
property float32 nz
property uchar red
property uchar green
property uchar blue
element face {num_faces}
property list uchar int vertex_indices
end_header
"""

# データ読み込みと変換
vertices = []
faces = []
with open(csv_file, 'r') as f:
    lines = f.read().splitlines()
    num_vertices = len(lines)
    num_faces = 0

    for line in lines:
        parts = line.split(',')
        if len(parts) == 9:
            x, y, z, r, g, b, nx, ny, nz = map(float, parts)
            vertices.append((x, y, z, nx, ny, nz, r, g, b))
        elif len(parts) == 3:
            num_faces += 1

# PLYファイルへ書き込み
with open(ply_file, 'w') as f:
    f.write(header.format(num_vertices=num_vertices, num_faces=num_faces))
    for vertex in vertices:
        f.write(" ".join(map(str, vertex)) + "\n")

print(f"PLYファイル {ply_file} に点群データが保存されました。")
