import os
import matplotlib.pyplot as plt
from PIL import Image

# PNG画像が格納されたディレクトリ
directory = '/home/sowa/prog/pointcloud/picture/eye_rgbhsv'

# ディレクトリ内のPNGファイルを取得
png_files = [file for file in os.listdir(directory) if file.endswith('.png')]

# 画像を表示するためのサブプロットを作成
num_images = len(png_files)
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

# 各画像を表示
for i, file_name in enumerate(png_files):
    img = Image.open(os.path.join(directory, file_name))
    axes[i].imshow(img)
    axes[i].axis('off')  # 軸を非表示にする
    axes[i].set_title(file_name)

plt.tight_layout()
plt.show()
