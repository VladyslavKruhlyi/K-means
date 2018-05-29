from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'x': [22, 30, 38, 28, 39, 43, 34, 55, 55, 62, 61, 62, 65, 63, 65, 71, 74, 79, 82],
    'y': [29, 16, 10, 42, 64, 36, 45, 89, 53, 60, 46, 33, 48, 33, 24, 18, 39, 71, 14]
})
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y'}
colors = list(map(lambda x: colmap[x+1], labels))
plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
