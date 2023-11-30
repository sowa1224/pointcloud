import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

# グレースケール画像の生成
# この例では、ランダムな点群データを仮定します
# 実際のアプリケーションでは、点群データを適切な方法でグレースケール画像に変換する必要があります
point_cloud_data = np.random.randint(0, 255, (500, 500), dtype=np.uint8)

# グレースケール画像を前処理
# グレースケール画像をWatershedセグメンテーションに適した形式に変換します
# ここでは、単純に二値化を行いますが、実際のアプリケーションに合わせて前処理を調整する必要があります
_, binary_image = cv2.threshold(point_cloud_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Watershedセグメンテーションを適用
markers = cv2.connectedComponents(binary_image)[1]
markers = markers + 1
markers[binary_image == 0] = 0
segmented_image = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
segmented_image = segmented_image.astype(np.uint8)

# セグメンテーション結果を表示
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# サンプルの点群データ (x, y, z, r, g, b)
points = np.array([
    [1, 2, 3, 225, 231, 57],  # 赤
    [4, 5, 6, 53, 243, 89],  # 緑
    [7, 8, 9, 0, 0, 255],  # 青
])

# 新しい3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 各点の座標と色情報をプロット
for point in points:
    x, y, z, r, g, b = point
    ax.scatter(x, y, z, c=(r/255, g/255, b/255), marker='o')
    ax.text(x, y, z, f'({x}, {y}, {z})', fontsize=10, color='black')

# 軸ラベル
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# グラフを表示
plt.show()
