import open3d as o3d

# Loading point cloud
print("Loading point cloud")
# PLYファイルのパス　
ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/bottle.ply"
# PLYファイルの読み込み
ptCloud= o3d.io.read_point_cloud(ply_file)

# RGBカラー情報を持つポイントクラウドの表示
o3d.visualization.draw_geometries([ptCloud])

# デプス情報のみを持つポイントクラウドの作成
depth_point_cloud = o3d.geometry.PointCloud()
depth_point_cloud.points = ptCloud.points
depth_point_cloud.colors = o3d.utility.Vector3dVector([[1, 1, 1]] * len(ptCloud.points))  # 白色で塗りつぶす

# デプス情報のみを表示
o3d.visualization.draw_geometries([depth_point_cloud])