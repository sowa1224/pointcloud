import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()


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

# 色情報を保存するためのオプション
ply.set_option(rs.save_to_ply.option_ply_texture, True)

# PLYファイルを保存
ply.process(depth_frame)


# PLYファイルを表示
pcd = o3d.io.read_point_cloud("/home/sowa/prog/pointcloud/pointcloud_ply_data/potato_1011.ply")
o3d.visualization.draw_geometries([pcd])

