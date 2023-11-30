import cv2
import pyrealsense2 as rs
import numpy as np
import os

# RealSenseパイプラインの設定python3 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
# 保存先ディレクトリの指定
save_directory_RGB = '/home/sowa/prog/pointcloud/RGB,DEPTH/RGB/'  # 保存したいディレクトリのパスを指定してください
save_directory_DEPTH = '/home/sowa/prog/pointcloud/RGB,DEPTH/DEPTH/'  # 保存したいディレクトリのパスを指定してください

# パイプラインを開始する
profile = pipeline.start(config)

try:
    frame_count = 0
    while True:
        # フレームを待ち、デプスとカラーのフレームを取得する
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # フレームが存在する場合
        if depth_frame and color_frame:
            # デプス画像の処理
            depth_image = np.asanyarray(depth_frame.get_data())

            # カラー画像の処理
            color_image = np.asanyarray(color_frame.get_data())


            # ファイルパスの設定
            depth_filename = os.path.join(save_directory_DEPTH, f'depth_{frame_count}.png')
            color_filename = os.path.join(save_directory_RGB, f'color_{frame_count}.png')

            # デプス画像を保存
            cv2.imwrite(depth_filename, depth_image)

            # カラー画像を保存
            cv2.imwrite(color_filename, color_image)

            print(f'Saved {depth_filename} and {color_filename}')
            frame_count += 1


            data_filename =  os.path.join(save_directory_RGB,f'data_{frame_count}.csv')
            with open(data_filename, 'w') as f:
                for y in range(color_image.shape[0]):
                    for x in range(color_image.shape[1]):
                        b, g, r = color_image[y, x]
                        f.write(f'Pixel ({x}, {y}): R={r}, G={g}, B={b}\n')


except KeyboardInterrupt:
    pass
finally:
    # パイプラインを停止し、リソースを解放する
    pipeline.stop()
