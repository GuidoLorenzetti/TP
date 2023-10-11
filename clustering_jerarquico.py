import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Cargar el dataset
df = pd.read_csv('Crop_recommendation.csv')

# Convierte la columna 'label' en una columna categórica
df['label'] = df['label'].astype('category')

# Selecciona las características numéricas
xWheat = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Escala las características
xWheat_scaled = StandardScaler().fit_transform(xWheat)

# Creamos el dendograma para encontrar el número óptimo de clusters
dendrogram =sch.dendrogram(sch.linkage(xWheat_scaled, method ='ward'))
plt.title('Dendograma')
plt.xlabel('Categorias')
plt.ylabel('Distancias Euclidianas')
plt.show()

# Ajustando Clustering Jerárquico al conjunto de datos
hc = AgglomerativeClustering(n_clusters = 9,
 metric= 'euclidean',
 linkage = 'ward')
y_hc = hc.fit_predict(xWheat_scaled)
plt.scatter(xWheat_scaled[y_hc == 0, 0], xWheat_scaled[y_hc == 0, 1],
s = 100, c = 'pink', label = 'Cluster 1')
plt.scatter(xWheat_scaled[y_hc == 1, 0], xWheat_scaled[y_hc == 1, 1],
s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(xWheat_scaled[y_hc == 2, 0], xWheat_scaled[y_hc == 2, 1],
s = 100, c = 'black', label = 'Cluster 3')
plt.scatter(xWheat_scaled[y_hc == 3, 0], xWheat_scaled[y_hc == 3, 1],
s = 100, c = 'grey', label = 'Cluster 4')
plt.scatter(xWheat_scaled[y_hc == 4, 0], xWheat_scaled[y_hc == 4, 1],
s = 100, c = 'orange', label = 'Cluster 5')
plt.scatter(xWheat_scaled[y_hc == 5, 0], xWheat_scaled[y_hc == 5, 1],
s = 100, c = 'purple', label = 'Cluster 6')

plt.scatter(xWheat_scaled[y_hc == 6, 0], xWheat_scaled[y_hc == 6, 1],
s = 100, c = 'brown', label = 'Cluster 7')
plt.scatter(xWheat_scaled[y_hc == 7, 0], xWheat_scaled[y_hc == 7, 1],
s = 100, c = 'yellow', label = 'Cluster 8')
plt.scatter(xWheat_scaled[y_hc == 8, 0], xWheat_scaled[y_hc == 8, 1],
s = 100, c = 'green', label = 'Cluster 9')

plt.title('Clusters of customers')
plt.xlabel('Perimeter')
plt.ylabel('Category')
plt.legend()
plt.show()


from gap_statistic import OptimalK

""" K-means """
gs_obj = OptimalK(n_jobs=1, n_iter= 10)
n_clusters = gs_obj(xWheat_scaled, n_refs=50, cluster_array=np.arange(1, 15))
print('Optimal number of clusters: ', n_clusters)

""" Clustering Jerárquico """
gs_obj = OptimalK(n_jobs=1, n_iter=20)
n_clusters = gs_obj(xWheat_scaled.astype('float'), n_refs=60,
cluster_array=np.arange(2, 10))
print('Optimal number of clusters: ', n_clusters)

print(f'Silhouette Score(n=4): {silhouette_score(xWheat_scaled, y_hc)}')
