import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('Crop_recommendation_corrected.csv')
df['label']=df['label'].astype('category')

xWines = df.drop(['label'], axis=1).drop(['colors'], axis=1)
yWines = df['label']

xWines_scaled = StandardScaler().fit_transform(xWines)

pca = PCA(n_components=7)
pca_Features=pca.fit(xWines_scaled)

# Bar plot of explained_variance
plt.bar(range(1,len(pca.explained_variance_)+1),
pca.explained_variance_)
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red', label='Cumulative Explained Variance')
plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')
plt.show()

pca = PCA(n_components=3)
pca_Features=pca.fit_transform(xWines_scaled)

pcaWines = pd.DataFrame(data=pca_Features, columns=['PC1', 'PC2', 'PC3'])
pcaWines['label'] = yWines
pcaWines['colors'] = df['colors']

# Configura el estilo de Seaborn
sns.set(style="whitegrid")

# Crea una figura y un eje 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Define las variables para los ejes x, y y z
x = pcaWines['PC1']
y = pcaWines['PC2']
z = pcaWines['PC3']

# Mapea los colores a las etiquetas en la columna 'label'
colors = pcaWines['colors']
print(colors)
# Crea un gráfico de dispersión 3D
scatter = ax.scatter(x, y, z, c=colors, marker='o', label='3D PCA Scatter Plot')

# Configura etiquetas de los ejes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Añade una leyenda
legend = ax.legend(*scatter.legend_elements(), title='Categories')
ax.add_artist(legend)

# Muestra el gráfico
plt.show()