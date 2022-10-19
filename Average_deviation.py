import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sample_create
from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False
         }


def km_deviation(Ks, datas):
    meanDispersions = []
    cluster_center = {}
    cluster_labels = {}
    for k in Ks:
        km = KMeans(n_clusters=k,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)

        km.fit(datas)
        cluster_center[k] = km.cluster_centers_
        cluster_labels[k] = km.labels_
        meanDispersions.append(sum(
            np.min(cdist(datas, km.cluster_centers_, 'euclidean'), axis=1)) / datas.shape[0])

    plt.rcParams['font.sans-serif'] = 'SimHei'
    # plt.figure(facecolor='lightyellow')

    # Draw a line graph of the sum of squares of deviations corresponding to different super parameters K
    plt.plot(Ks, meanDispersions, 'bo-', mfc='r')
    plt.xlabel('Number of cluster centers k', fontsize=14)
    plt.ylabel('Sum of the Squared Errors', fontsize=14)
    plt.title('The Optimal Number of Clusters in K-means', fontsize=14)
    plt.tick_params(labelsize=13)

    plt.show()
    return cluster_center, cluster_labels


def cluster_center_plot(datas, center):
    x = datas[:, 0]
    y = datas[:, 1]
    plt.figure(2)
    label = ["Section center", "Gaze Point"]
    plt.scatter(x, y, c='b', marker='.', s=20)
    plt.plot(center[:, 0], center[:, 1], 'ro--')
    plt.legend(label, loc=0, ncol=1)
    plt.xlabel('X-axis(Normalized coordinate)')
    plt.ylabel('Y-axis(Normalized coordinate)')
    plt.title('Number of eye movement sequence segments N = {}'.format(len(center)))
    plt.show()


def Remove_redundant_data(labels, centers):
    temp_list = func(labels)
    n = len(temp_list)
    center_order = np.empty(shape=[n, 2])
    for i in range(0, n):
        j = temp_list[i]
        center_order[i, :] = centers[j, :]
    return center_order


def func(labels):
    temp_list = []
    for i in labels:
        if i not in temp_list:
            temp_list.append(i)
    return temp_list


if __name__ == '__main__':
    # Create a simulation cluster dataset
    datas = sample_create.txt2matrix('collect_clasi_train.txt')
    rcParams.update(config)
    datas = datas[:, 0:2]
    Ks = range(1, 10)
    center, labels = km_deviation(Ks, datas)
    center_order = Remove_redundant_data(labels[4], center[4])
    cluster_center_plot(datas, center_order)
    print("------------------end---------------------")
