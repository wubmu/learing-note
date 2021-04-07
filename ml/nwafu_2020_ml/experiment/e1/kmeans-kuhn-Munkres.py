from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances_argmin
from time import time
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from munkres import Munkres, print_matrix
#######################################################################
# 1. 读取数据
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataMat = loadmat('datasets/MNIST.mat')
lungData = loadmat('datasets/lung.mat')
YaleData = loadmat('datasets/Yale.mat')
feature = dataMat['X']
labels_true = dataMat['Y'].flatten()
# (3000, 784)
# (3000, 1)
# print(feature[0])
# print(label.shape)
########################################################################
# 2.kmeans聚类

k_means = KMeans(n_clusters=10 , n_init=10)
t0 = time()
res = k_means.fit(feature)
t_batch = time() - t0
print(res)

'''
reduced_data = PCA(n_components=2).fit_transform(feature)
k_means = KMeans(n_clusters=10 , n_init=10)
t0 = time.time()
res = k_means.fit(feature)
t_batch = time.time() - t0
print(res)
'''
########################################################################
# 3. 画图

# plt.style.use({'figure.figsize': (5, 5)})
col = 4
# fig = plt.figure(figsize=(8 ,3))
for i in range(10):
    # print('当前聚类中心：', k_means.cluster_centers_[i])
    plt.subplot(3, 4, i+1)
    plt.imshow(k_means.cluster_centers_[i].reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.show()
########################################################################
# 轮廓系数
# labels_pred = k_means.predict(feature)
# 评估指标——轮廓系数,前者为所有点的平均轮廓系数，后者返回每个点的轮廓系数
# from sklearn.metrics import silhouette_score, silhouette_samples
# print("轮廓系数：",silhouette_score(feature, labels_pred))
# print("轮廓系数：",silhouette_score(feature, labels_true))

#######################################################################
# 标签匹配
def best_map(L1, L2):
    # L1是真实标签，L2是聚类标签
    Label1 = np.unique(L1) #标签去重
    nClass1 = len(Label1) # 标签的大小
    Label2 = np.unique(L2)  # 标签去重
    nClass2 = len(Label2)  # 标签的大小
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1) # 计算两个集合的交集 np.intersect1d(x,y))
            # [1,1,0,0] *[1,0,1,0] = [1,0,0,0] sum-> 1
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)

    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    # 标签替换 如果L2 == labels[i]  替换成 Label1[c[i]]
    '''
    c = [5 3 4 1 6 7 9 0 8 2] 对应的标签编号
    for i in len(nClass2)
        选取符合第i个类别的样本 L2==Label2
        替换第2个类别对应的标签 newL2[L2 == Label2] = label1[c[i]]
    '''
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2
# best_map(labels_true.flatten(),labels_pred) #(3000,1) ->(3000,) (3000,)
######################################################################################
# 计算纠正过后的准确率
def acc(true_labels, pred_labels):
    mk_pred_labels = best_map(true_labels, pred_labels)
    sum = np.sum(true_labels == mk_pred_labels)
    return sum/len(true_labels)

# print(acc(labels_true,labels_pred))
###################################################################################
#
def bench_kmeans(kmeans, name):
    # 读取数据
    dataMat = loadmat('datasets/'+name+'.mat')
    # lungData = loadmat('datasets/lung.mat')
    # YaleData = loadmat('datasets/Yale.mat')
    feature = dataMat['X']
    labels_true = dataMat['Y'].flatten()

    (n_samples, n_features), n_classes= feature.shape, np.unique(labels_true).size

    # print(
    #     f"# digits: {n_classes}; # samples: {n_samples}; # features {n_features}"
    # )


    t0 = time()
    estimator = make_pipeline(kmeans).fit(feature)
    fit_time = time() - t0
    labels_pred = estimator[-1].predict(feature)
    # 样本离最近聚类中心的总和
    results = [name, fit_time]

    # 标签匹配
    labels_mk = best_map(labels_true, labels_pred)
    # Homogeneity, completeness and V-measure（聚类数量情况）
    '''
    同质性homogeneity：每个群集只包含单个类的成员。
    完整性completeness：给定类的所有成员都分配给同一个群集。
    两者的调和平均V-measure：
    adjusted_rand_score： 调整后的兰德指数
    AMI:调整互信息
    MI:互信息
    NMI: 标准互信息
    '''
    clustering_metrics = {
        metrics.normalized_mutual_info_score,
        metrics.accuracy_score
    }

    results += [m(labels_true, labels_mk) for m in clustering_metrics]

    # 计算轮廓系数,
    # results += [
    #     metrics.silhouette_score(feature, labels_mk,
    #                              metric="euclidean")
    # ]

    # 输出结果
    formatter_result = ("{:9s}\t{:.3f}s\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))
############################################################################
# Kmeans运行基准
print(50*'-')
print("数据集\t\ttime\tNMI\tAcc")
kmeans = KMeans(init="k-means++", n_clusters=10)
bench_kmeans(kmeans,'MNIST')

kmeans = KMeans(init="k-means++", n_clusters=5)
bench_kmeans(kmeans,'lung')

kmeans = KMeans(init="k-means++", n_clusters=15)
bench_kmeans(kmeans,'Yale')

print(50*'-')
print("高斯模型")
gmm = GaussianMixture(n_components=10)
bench_kmeans(gmm,'MNIST')

gmm = GaussianMixture(n_components=5)
bench_kmeans(gmm,'lung')

gmm = GaussianMixture(n_components=15)
bench_kmeans(gmm,'Yale')
