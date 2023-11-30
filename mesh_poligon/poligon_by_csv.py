import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import time

# 開始時間を記録
start_time = time.time()
# CSVファイルから点群データを読み込みます
def load_point_cloud_from_csv(file_path, skip_header=True):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        if skip_header:
            next(csv_reader)  # ヘッダーをスキップ
        
        data = []
        
        for row in csv_reader:
            # CSVファイルからXYZ座標を読み込む
            xyz = [float(row[i]) for i in range(3)]  # XYZ座標
            normals = [float(row[i]) for i in range(6, 9)]  # 法線情報 (nx, ny, nz)
            data.append(xyz + normals)
        
        data = np.array(data)  # リストからNumPy配列に変換
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])  # XYZ座標
        point_cloud.normals = o3d.utility.Vector3dVector(data[:, 3:])  # 法線情報

    return point_cloud


# ポリゴンモデルを生成し、3Dで表示
def generate_and_visualize_mesh(point_cloud):
    # ボールピボットを使用してポリゴンモデルを生成
    radius = 0.007  # ボールピボットの半径
    angle = 15  # ピボットの角度（度数法）
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector([radius, angle])
    )

    # 3Dビューアーを開き、ポリゴンモデルを表示
    o3d.visualization.draw_geometries([mesh])

    return mesh  # 生成したメッシュを返す




if __name__ == "__main__":
    file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1025.csv"
    #file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1017.csv"
    point_cloud = load_point_cloud_from_csv(file_path)
    mesh = generate_and_visualize_mesh(point_cloud)
    # ポリゴンの数を取得
    num_polygons = len(mesh.triangles)

    # 各ポリゴンの中心座標を計算
    polygon_centers = []
    for triangle in mesh.triangles:
        vertex1 = np.array(mesh.vertices[triangle[0]])
        vertex2 = np.array(mesh.vertices[triangle[1]])
        vertex3 = np.array(mesh.vertices[triangle[2]])
        center = (vertex1 + vertex2 + vertex3) / 3
        polygon_centers.append(center)

        # 終了時間を記録
    end_time = time.time()


    # 0.04から一定の範囲内にあるポリゴンの中心座標を赤色でプロット
    red_polygon_centers = []
    red_normals = []
    for i, center in enumerate(polygon_centers):
#        if -0.1 <= center[0] <= -0.099:  # 0.04から±0.01の範囲を指定
        if 0.019 <= center[0] <= 0.02:  # 0.04から±0.01の範囲を指定
            red_polygon_centers.append(center)
            red_normals.append(mesh.triangle_normals[i])  # ポリゴンの法線ベクトルを取得


    # 条件を満たすポリゴンの中心座標だけを赤色でプロット
    x_red = [center[0] for center in red_polygon_centers]
    y_red = [center[1] for center in red_polygon_centers]
    z_red = [center[2] for center in red_polygon_centers]

    red_polygon_centers = np.array(red_polygon_centers)
    red_normals = np.array(red_normals)
    red_combined = np.hstack((red_polygon_centers, red_normals))
    odd_elements = red_combined[1::2]

    # すべてのポリゴンの中心座標を青色でプロット
    x_blue = [center[0] for center in polygon_centers]
    y_blue = [center[1] for center in polygon_centers]
    z_blue = [center[2] for center in polygon_centers]

    fig = plt.figure(figsize=(10, 8))  # 幅10インチ、高さ8インチ
    ax = fig.add_subplot(111, projection='3d')
  #  ax.scatter(x_blue, y_blue, z_blue, c='b', marker='o', label='blue')
    ax.scatter(x_red, y_red, z_red, c='r', marker='o', label='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('poligon')
    plt.show()
    print(red_combined)
    print(np.size(red_combined))
    

    #ポリゴン数の結果
    print(f"すべてのポリゴン数: {len(x_blue)}")
    print(f"一つの軌道用のポリゴン数: {len(x_red)}")
    print(f"抽出した点の個数：{len(odd_elements)}")
    # 経過時間を計算
    elapsed_time = end_time - start_time
    print(f"コードの実行にかかった時間: {elapsed_time}秒")
    o3d.io.write_triangle_mesh("/home/sowa/prog/pointcloud/poligon/output_1025.ply", mesh, write_ascii=True)