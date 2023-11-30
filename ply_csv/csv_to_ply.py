import csv

csv_file_path = "/home/sowa/prog/pointcloud/csv/potato_filtered_1011.csv"
ply_file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/csv_to_ply_test_1015.ply"


# CSVファイルを読み込む
with open(csv_file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)  # ヘッダー行をスキップ
    
    # データをリストとして読み込む
    data = [row for row in reader]

# PLYファイルに書き出し
with open(ply_file_path, 'w') as ply_file:
    # ヘッダー情報を書き出し
    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('comment pointcloud converted from CSV\n')
    ply_file.write(f'element vertex {len(data)}\n')
    ply_file.write('property float32 x\n')
    ply_file.write('property float32 y\n')
    ply_file.write('property float32 z\n')
    ply_file.write('property uchar red\n')
    ply_file.write('property uchar green\n')
    ply_file.write('property uchar blue\n')
    ply_file.write('property float32 nx\n')
    ply_file.write('property float32 ny\n')
    ply_file.write('property float32 nz\n')
    ply_file.write('end_header\n')
    
    # データを書き出し
    for row in data:
        x, y, z, r, g, b, nx, ny, nz = row
        ply_file.write(f'{x} {y} {z} {r} {g} {b} {nx} {ny} {nz}\n')

print('PLYファイルの変換が完了しました。')
