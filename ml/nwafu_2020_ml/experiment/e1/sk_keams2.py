import numpy as np
from scipy.io import loadmat


##############################################################################
# 1. 数据集准备
from sklearn.preprocessing import StandardScaler

dataMat = loadmat('datasets/MNIST.mat')
print(dataMat.keys())
feature = dataMat['X']
labels_true = dataMat['Y'].flatten()
(n_samples, n_features), n_digits = feature.shape, np.unique(labels_true).size

print(
    f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}"
)

###############################################################################
# 2. 定义我们的评估基准
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline


def bench_k_means(kmeans, name, data, labels):
    # 初始化方法不一样
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name,fit_time, estimator[-1].inertia_]

    # 定义估计量的评价指标

    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
        metrics.normalized_mutual_info_score,
        metrics.accuracy_score
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # 计算轮廓系数 需要所有的数据集
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]
    # 输出结果
    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))

#################################################################################
#3. 运行基准
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tNMI\t\tacc\t\tsilhouette')

kmeans = KMeans(init="k-means++", n_clusters=n_digits,random_state=1)
bench_k_means(kmeans=kmeans, name="k-means++", data = feature, labels=labels_true)

kmeans = KMeans(init="random", n_clusters=n_digits, random_state=1)
bench_k_means(kmeans=kmeans, name="random", data = feature, labels=labels_true)

pca = PCA(n_components=n_digits).fit(feature)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=feature, labels=labels_true)

print(82 * '_')

#####################################################################################
# 4, 可视化结果在PCA降维数据
import matplotlib.pyplot as plt
