import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)  # RGBストリームの設定

# パイプラインを開始
pipeline.start(config)

# CSVファイルに保存するための空のリストを作成
pixel_data = []

try:
    while True:
        # フレームの取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # フレームデータをnumpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())

        # カラーフレームを表示
        cv2.imshow('Color Stream', color_image)

        # クリックで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 各ピクセルのRGBと座標を取得してリストに追加
        for y in range(color_image.shape[0]):
            for x in range(color_image.shape[1]):
                b, g, r = color_image[y, x]  # 各ピクセルのBGR値を取得
                pixel_data.append([x, y, r, g, b])  # 座標とRGB値をリストに追加

finally:
    # CSVファイルに保存
    df = pd.DataFrame(pixel_data, columns=['X', 'Y', 'R', 'G', 'B'])
    df.to_csv('pixel_data.csv', index=False)

    # パイプラインを停止
    pipeline.stop()

    # カラーストリームのウィンドウを閉じる
    cv2.destroyAllWindows()
