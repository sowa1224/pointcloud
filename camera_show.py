import cv2
import pyrealsense2 as rs
import numpy as np
# カメラの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # フレームの取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # フレームが取得できなかった場合はスキップ
        if not color_frame:
            continue

        # フレームデータをOpenCVの画像形式に変換
        color_image = np.asanyarray(color_frame.get_data())

        # 画像の表示
        cv2.imshow('Color Image', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # カメラの停止とウィンドウの破棄
    pipeline.stop()
    cv2.destroyAllWindows()