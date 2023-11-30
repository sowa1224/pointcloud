import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# RealSenseパイプラインを初期化
pipe = rs.pipeline()
config = rs.config()

# 深度ストリームを有効にする
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# RGBストリームを有効にする
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# RealSenseカメラを開始
pipe.start(config)

# ポイントクラウドオブジェクトを初期化
pc = rs.pointcloud()

# カメラからフレームを待機
frames = pipe.wait_for_frames()

# 深度データを取得
depth_frame = frames.get_depth_frame()

# RGBデータを取得
color_frame = frames.get_color_frame()
color_data = color_frame.get_data()

# ポイントクラウドを生成
pc.map_to(color_frame)
points = pc.calculate(depth_frame)

# PLYファイルへの保存オプションを設定
ply = rs.save_to_ply("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1011.ply")

# オプションを設定（バイナリ形式、法線、色情報）
ply.set_option(rs.save_to_ply.option_ply_binary, True)
ply.set_option(rs.save_to_ply.option_ply_normals, True)


# PLYファイルを保存
ply.process(depth_frame)


# PLYファイルを表示
pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1011.ply")
o3d.visualization.draw_geometries([pcd])

