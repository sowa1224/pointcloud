import open3d as o3d
import colorsys

# PLYファイルの読み込み
pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_peeling.ply")
#pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017.ply")

# 抽出したいHueの範囲を指定 (例: 15から40)
min_hue = 20
max_hue = 70
# Hue範囲内の点を抽出
selected_points = []
selected_colors = []
selected_normals = []
for i, point in enumerate(pcd.points):
    color = pcd.colors[i]
    # RGBをHSVに変換
    hsv_color = colorsys.rgb_to_hsv(color[0], color[1], color[2])
    hue = hsv_color[0] * 360  # 色相を0から360度の範囲に変換
    if min_hue <= hue <= max_hue:
        selected_points.append(point)
        selected_colors.append(color)
        selected_normals.append(pcd.normals[i])

# 新しいPLYファイルに抽出された点を保存（x, y, z, r, g, b, nx, ny, nz を含む）
selected_pcd = o3d.geometry.PointCloud()
selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
selected_pcd.colors = o3d.utility.Vector3dVector(selected_colors)
selected_pcd.normals = o3d.utility.Vector3dVector(selected_normals)


#o3d.io.write_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_ver1.ply", selected_pcd, write_ascii=True)
o3d.io.write_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_peeling_ver1.ply", selected_pcd, write_ascii=True)
#file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_ver1.ply"
file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_peeling_ver1.ply"

point_cloud = o3d.io.read_point_cloud(file_path)



# ポイントクラウドを表示する
o3d.visualization.draw_geometries([point_cloud])