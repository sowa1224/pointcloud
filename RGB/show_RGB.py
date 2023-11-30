import cv2
import os
import numpy as np
# CSVファイルが格納されたディレクトリのパス
directory_path = "/home/sowa/prog/pointcloud/RGB,DEPTH/RGB/"

# ディレクトリ内のすべてのCSVファイルを読み込む
file_list = os.listdir(directory_path)
csv_files = [file for file in file_list if file.endswith(".csv")]

# カメラの解像度
width, height = 1920, 1080

# マウスクリックイベントを処理する関数
def on_mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        virtual_x, virtual_y = get_virtual_coordinates(x, y)
        print(f"Clicked on {csv_file}: Pixel ({x}, {y}) corresponds to virtual coordinates ({virtual_x}, {virtual_y})")


# CSVファイルごとに処理
for csv_file in csv_files:
    # CSVファイルのパス
    file_path = os.path.join(directory_path, csv_file)

    # CSVファイルを開いてピクセル情報を取得
    pixel_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Pixel'):
                # 'Pixel'から始まる行を対象にする場合
                values = line.split(': ')[1].split(', ')
                r, g, b = map(int, [value.split('=')[1] for value in values])
                pixel_data.append([r, g, b])
    # 画像の作成
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for i, pixel in enumerate(pixel_data):
        x = i % width  # 画像内のx座標
        y = i // width  # 画像内のy座標

        if x < width and y < height:  # 画像内にピクセルがあるか確認
            image[y, x] = pixel
    # 画像の表示
    cv2.imshow(f'Image from {csv_file}', image)

    # マウスクリックイベントを処理する関数
    def on_mouse_click(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked on {csv_file}: Pixel ({x}, {y}): R={image[y, x][2]}, G={image[y, x][1]}, B={image[y, x][0]}")

    # マウスクリックイベントのバインディング
    cv2.setMouseCallback(f'Image from {csv_file}', on_mouse_click)

# キー入力待ち（すべての画像表示が終わるまで）
cv2.waitKey(0)
cv2.destroyAllWindows()

# カメラの解像度
camera_width, camera_height = 1920, 1080


