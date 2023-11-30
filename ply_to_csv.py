import open3d as o3d
import numpy as np
import pandas as pd
import colorsys
import os

# Directory containing PLY files
ply_directory = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1118/unpeel/"

# Output CSV directory
csv_directory = "/home/sowa/prog/pointcloud/csv/potato_1118/unpeel/"

# Iterate through all PLY files in the directory
for file_name in os.listdir(ply_directory):
    if file_name.endswith(".ply"):
        # Construct the full path for PLY and CSV files
        ply_file_path = os.path.join(ply_directory, file_name)
        csv_file_path = os.path.join(csv_directory, file_name.replace(".ply", ".csv"))

        # Load PLY file
        ptCloud = o3d.io.read_point_cloud(ply_file_path)

        # Extract point cloud data
        points = np.asarray(ptCloud.points)
        colors = np.asarray(ptCloud.colors)
        normals = np.asarray(ptCloud.normals)

        # 深度が5メートル以内かつ法線ベクトルが[0, 0, 1]以外の点群を抽出
        selected_points = points
        selected_colors = colors
        selected_normals = normals

# RGBカラーをHSVに変換（0から1の範囲）
        hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in selected_colors])

# HSVを0から255の範囲に変換
        hsv_colors_255 = (hsv_colors * 255).astype(int)


        # Your existing point cloud processing code...

        # Convert to DataFrame
        df = pd.DataFrame({
            'X': points[:, 0],
            'Y': points[:, 1],
            'Z': points[:, 2],
            'R': colors[:, 0],
            'G': colors[:, 1],
            'B': colors[:, 2],
            'nX': normals[:, 0],
            'nY': normals[:, 1],
            'nZ': normals[:, 2],
            'H': hsv_colors_255[:, 0],
            'S': hsv_colors_255[:, 1],
            'V': hsv_colors_255[:, 2]
        })

        # Save CSV file
        df.to_csv(csv_file_path, index=False, header=False)

        print(f"Converted {file_name} to {csv_file_path}")

print("Conversion of PLY files to CSV files completed.")

""""
#ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1113/unpeel/potato_unpeel_1113_0.ply"
ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1113/peel/potato_peel_1113_0.ply"
#ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_unpeel_4.ply"
# PLYファイルの読み込み
ptCloud = o3d.io.read_point_cloud(ply_file)

# 点群データをnumpy配列として取得
points = np.asarray(ptCloud.points)
colors = np.asarray(ptCloud.colors)
normals = np.asarray(ptCloud.normals)

# 深度が5メートル以内かつ法線ベクトルが[0, 0, 1]以外の点群を抽出
selected_points = points
selected_colors = colors
selected_normals = normals

# RGBカラーをHSVに変換（0から1の範囲）
hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in selected_colors])

# HSVを0から255の範囲に変換
hsv_colors_255 = (hsv_colors * 255).astype(int)

# 抽出した点群データをDataFrameに変換
df = pd.DataFrame({
    'X': selected_points[:, 0],
    'Y': selected_points[:, 1],
    'Z': selected_points[:, 2],
    'R': selected_colors[:, 0],
    'G': selected_colors[:, 1],
    'B': selected_colors[:, 2],
    'nX': selected_normals[:, 0],
    'nY': selected_normals[:, 1],
    'nZ': selected_normals[:, 2],
    'H': hsv_colors_255[:, 0],  # 追加: 色相
    'S': hsv_colors_255[:, 1],  # 追加: 彩度
    'V': hsv_colors_255[:, 2]   # 追加: 明度
})



# CSVファイルとして保存（ヘッダー行を含めて保存）
df.to_csv('/home/sowa/prog/pointcloud/csv/potato_1113/peel/potato_peel_1113_0.csv', index=False, header=False)  # headerをTrueに変更
#df.to_csv('/home/sowa/prog/pointcloud/csv/potato_1113/unpeel/potato_unpeel_1113_0.csv', index=False, header=False)  # headerをTrueに変更
print("点群データがCSVファイルに保存されました。")
print("CSVファイルの大きさは", df.shape)
"""