import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd

# データ生成（サンプルデータを使う場合、ここにデータを読み込んでください）
# data = np.array([...])
# CSVファイルからデータを読み込む
csv_file_path = "/home/sowa/prog/pointcloud/csv/extracted_data1025.csv"
data = pd.read_csv(csv_file_path)



# X-meansの実行
def xmeans(data, kmax):
    bic_values = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        bic = kmeans.inertia_
        bic_values.append(bic)

    x = np.arange(1, kmax + 1)
    plt.plot(x, bic_values, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("BIC")
    plt.title("X-means BIC Plot")
    plt.show()

    best_k = np.argmin(bic_values) + 1
    print("Best number of clusters (X-means):", best_k)

    best_model = KMeans(n_clusters=best_k, random_state=0).fit(data)
    return best_model

# クラスタリング実行
kmax = 10  # 最大クラスタ数
best_kmeans = xmeans(data, kmax)

# 結果の可視化
# (クラスタリング結果を描画するコードをここに追加)

plt.scatter(data[:, 0], data[:, 1], c=best_kmeans.labels_)
plt.scatter(best_kmeans.cluster_centers_[:, 0], best_kmeans.cluster_centers_[:, 1], s=200, c='red')
plt.title("X-means Clustering Result")
plt.show()
