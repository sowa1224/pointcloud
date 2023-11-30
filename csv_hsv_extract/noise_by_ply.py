import open3d as o3d

# PLYファイルの読み込み
#pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_ver1.ply")
pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_peeling_ver1.ply")
# 半径外れ値除去を実行
cl, ind = pcd.remove_radius_outlier(nb_points=100, radius=0.008)
filtered_pcd = pcd.select_by_index(ind)

# 新しいPLYファイルに抽出された点を保存
#o3d.io.write_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_filtered_ver1.ply", filtered_pcd, write_ascii=True)

#file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_filtered_ver1.ply"

o3d.io.write_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_filtered_peeling_ver1.ply", filtered_pcd, write_ascii=True)

file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_viewer1017_filtered_peeling_ver1.ply"
point_cloud = o3d.io.read_point_cloud(file_path)

# ポイントクラウドを表示する
o3d.visualization.draw_geometries([point_cloud])
