from sklearn.cluster import KMeans


def apply_kmeans(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(features)
    return kmeans
