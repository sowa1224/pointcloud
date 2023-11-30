import open3d as o3d
import numpy as np
import csv

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

def generate_and_visualize_mesh(point_cloud):
    # ボールピボットを使用してポリゴンモデルを生成
    radius = 0.01  # ボールピボットの半径
    angle = 15  # ピボットの角度（度数法）
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector([radius, angle])
    )

    # 最大と最小のy座標を取得
    max_y = np.max(np.asarray(point_cloud.points)[:, 1])
    min_y = np.min(np.asarray(point_cloud.points)[:, 1])

    # 各ポリゴンと点の距離を計算
    distances = np.linalg.norm(np.asarray(mesh.vertices) - np.array([0, max_y, 0]), axis=1)

    # 最も近い2つのポリゴンのインデックスを取得
    nearest_polygon_indices = np.argsort(distances)[:2]

    # メッシュを複製して全てのポリゴンを灰色にする
    gray_mesh = mesh.clone()
    gray_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色

    # 最も近い2つのポリゴンを赤色にする
    for index in nearest_polygon_indices:
        gray_mesh.triangle_colors[index] = [1, 0, 0]

    # 3Dビューアーを開き、ポリゴンモデルを表示
    o3d.visualization.draw_geometries([gray_mesh])

    return gray_mesh

if __name__ == "__main":
    file_path = "/home/sowa/prog/pointcloud/csv/potato_hsv_eye_1017.csv"
    point_cloud = load_point_cloud_from_csv(file_path)
    generate_and_visualize_mesh(point_cloud)
