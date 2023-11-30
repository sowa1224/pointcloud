import open3d as o3d

#file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1113/unpeel/potato_unpeel_1113_0.ply"
file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1113/peel/potato_peel_1113_0.ply"
point_cloud = o3d.io.read_point_cloud(file_path)
# ポイントクラウドを表示する
o3d.visualization.draw_geometries([point_cloud])
