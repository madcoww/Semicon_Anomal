"""
Author : Wonjun Kim
e-mail : wonjun.kim@seculayer.com
Powered by Seculayer © 2024 AI Team, R&D Center.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Clustering:
    def __init__(self, df_sample, cluster_path, seed):
        self.df_sample = df_sample
        self.cluster_path = cluster_path
        self.seed = seed

    def sample_data(self, df, sample_size=10000):
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=self.seed)
        else:
            df_sample = df
        return df_sample

    def vectorize(self, df_sample):
        tfidf = TfidfVectorizer()
        X_sample = tfidf.fit_transform(df_sample['payload'])
        return X_sample, tfidf

    def reduce_dimensions(self, X_sample, n_components=100):
        svd = TruncatedSVD(n_components=n_components)
        X_reduced_sample = svd.fit_transform(X_sample)
        return X_reduced_sample, svd

    def find_best_kmeans(self, X_reduced_sample, n_clusters_range=(40, 51)):
        best_silhouette = -1
        best_n_clusters = 0
        best_kmeans = None
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1]):
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=self.seed)
            cluster_labels = kmeans.fit_predict(X_reduced_sample)
            silhouette_avg = silhouette_score(X_reduced_sample, cluster_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters
                best_kmeans = kmeans
        return best_kmeans, best_n_clusters, best_silhouette

    def apply_clustering(self, df, tfidf, svd, kmeans):
        X = tfidf.transform(df['payload'])
        X_reduced = svd.transform(X)
        df['cluster'] = kmeans.predict(X_reduced)
        return df

    def to_csv(self, df_clustered):
        df_clustered.to_csv(self.cluster_path)