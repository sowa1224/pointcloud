import open3d as o3d

# PLYファイルの読み込み
point_cloud = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_filtered_1017_peeling1_doublenoise.ply")

# ポリゴン化（ポワッシン表面再構築を使用）
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)

# ポリゴンモデルのエクスポート
o3d.io.write_triangle_mesh("/home/sowa/prog/pointcloud/poligon/test0.obj", mesh, write_ascii=True, compressed=False, cuda_hash=0)

mesh = o3d.io.read_triangle_mesh("/home/sowa/prog/pointcloud/poligon/test0.obj")
o3d.visualization.draw_geometries([mesh])


