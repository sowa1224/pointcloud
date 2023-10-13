import open3d as o3d
import numpy as np



if __name__ == "__main__":
    
    # Loading point cloud
    print("Loading point cloud")
    # PLYファイルのパス　
    ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/bottle.ply"
    # PLYファイルの読み込み
    ptCloud= o3d.io.read_point_cloud(ply_file)

    #抽出したポイントデータを新しいPLYファイルに保存
    output_ply_file = "/home/sowa/prog/pointcloud/pointcloud_ply_data/vertical.ply"


    # Estimation of normal vector of points
    ptCloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Visualization
    print("Vertical show")
    o3d.visualization.draw_geometries([ptCloud], point_show_normal=True)

    # Checking normal vectors
    print(np.asarray(ptCloud.normals))
    print(np.asarray(ptCloud.normals[0]))
    
    # Saving point cloud
    o3d.io.write_point_cloud(output_ply_file, ptCloud) #保存されるファイルの場所は15行目よりpointcloud_ply_dataのディレクトリ内になる


    #このままだた保存されるplyファイルはバイナリの形式になる、簡単に言うとバイナリではデータの中身が見えない、
    #よってバイナリからテキスト形式に変換する必要がある、以下がその手順になる



# ポイントクラウドをバイナリ形式で読み込む
point_cloud = o3d.io.read_point_cloud(output_ply_file)

# ポイントクラウドをテキスト形式に変換
text_data = o3d.io.write_point_cloud_to_string(point_cloud, write_ascii=True)

o3d.io.write_point_cloud(text_data, point_cloud, write_ascii=True)

# 新しいファイルにテキストデータを保存
text_file_path = "/home/sowa/prog/pointcloud/pointcloud_ply_data/vertical_text.ply"
with open(text_file_path, 'w') as file:
    file.write(text_data)