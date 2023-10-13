import open3d as o3d

file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato.ply"
point_cloud = o3d.io.read_point_cloud(file_path)



# ポイントクラウドを表示する
o3d.visualization.draw_geometries([point_cloud])
