import numpy as np
import cv2

file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1113/peel/potato_peel_1113_0.ply"

# PLYファイルの読み込み
with open(file_path, "r") as file:
    lines = file.readlines()

# データ部分の始まりを見つける
data_start = 0
for i, line in enumerate(lines):
    if "end_header" in line:
        data_start = i + 1
        break

# 点群データと色情報の取得
points = []
colors = []
for line in lines[data_start:]:
    elements = line.strip().split()
    if len(elements) == 6:  # 座標と色情報が含まれる行
        x, y, z = float(elements[0]), float(elements[1]), float(elements[2])
        r, g, b = int(elements[3]), int(elements[4]), int(elements[5])
        points.append([x, y, z])
        colors.append([r, g, b])

# 画像サイズの設定
width, height = 1920, 1080
image = np.zeros((height, width, 3), dtype=np.uint8)  # RGB画像用の配列

# 点群データをRGB画像にマッピング
for i in range(len(points)):
    x, y, z = points[i]
    r, g, b = colors[i]
    if 0 <= x < width and 0 <= y < height:  # 画像範囲内の点のみを処理
        image[int(y), int(x)] = [r, g, b]

# 画像の保存
cv2.imwrite("/home/sowa/prog/pointcloud/RGB,DEPTH/RGB/ply_rgb.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
