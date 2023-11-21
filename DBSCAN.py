import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import numpy as np

# Generate data
datasets = ['Gaussian Mixture dataset', 'Uniform Points dataset']
for no_db in range(len(datasets)):
    if no_db == 0:  # Gaussian Mixture dataset
        num_samples_total = 500
        cluster_centers = [(3, 3), (5, 7), (7, 3)]
        num_classes = len(cluster_centers)
        X, y = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes,
                          center_box=(0, 1),
                          cluster_std=0.5)
        np.save('./clusters.npy', X)
        X = np.load('./clusters.npy')
    elif no_db == 1:  # Uniform Points dataset
        num_samples_total = 250
        X, y = make_blobs(n_samples=num_samples_total, n_features=2, center_box=(0, 1),
                          cluster_std=15)
        np.save('./clusters.npy', X)
        X = np.load('./clusters.npy')

    # Compute DBSCAN for dataset
    epsilon = 1  # changeable
    minPoints = 4  # changeable
    db = DBSCAN(eps=epsilon, min_samples=minPoints).fit(X)
    labels = db.labels_

    no_clusters = len(np.unique(labels))
    no_noise = np.sum(np.array(labels) == -1, axis=0)
    print(datasets[no_db])
    print('Estimated number of clusters: %d' % no_clusters)
    print('Estimated number of noise points: %d' % no_noise)
    print('################################')
    # Generate scatter plot
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker="o", picker=True)
    plt.title(datasets[no_db])
    # plt.savefig('2.png')
    plt.show()
