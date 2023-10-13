import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

# RealSenseデバイスの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

# フレームの取得
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# RealSenseフレームからデータを抽出
depth_data = np.asanyarray(depth_frame.get_data())
color_data = np.asanyarray(color_frame.get_data())

# 3D点群データを作成
pcd = o3d.geometry.PointCloud.create_from_depth_image(
    o3d.geometry.Image(depth_data),
    o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
)

# カラーデータを追加
pcd.colors = o3d.utility.Vector3dVector(np.array(color_data).reshape(-1, 3) / 255.0)

# 法線ベクトルを計算
#o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# PLYファイルに保存
o3d.io.write_point_cloud("output.ply", pcd)

# RealSenseデバイスをシャットダウン
pipeline.stop()

# PLYファイルを表示
pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/output.ply")
o3d.visualization.draw_geometries([pcd])