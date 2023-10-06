import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Cargar el conjunto de datos desde un archivo CSV
df = pd.read_csv('Crop_recommendation.csv')

# Generar una paleta de colores única basada en las etiquetas de clase
palette = sns.color_palette('husl', n_colors=len(df['label'].unique()))

# Crear un diccionario para mapear etiquetas de clase a colores
color_dict = dict(zip(df['label'].unique(), palette))

# Aplicar el mapeo de colores a la columna 'label' y crear la columna 'Colores'
df['Colores'] = df['label'].map(color_dict)

# Convertir la columna 'label' en un tipo categórico
df['label'] = df['label'].astype('category')

# Seleccionar características numéricas y escalarlas
features = df.drop(['label', 'Colores'], axis=1)
scaled_features = StandardScaler().fit_transform(features)

# Aplicar t-SNE para reducir la dimensionalidad a 2D
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(scaled_features)

# Crear un DataFrame con etiquetas originales y resultados de t-SNE
tsne_df = pd.DataFrame({'label': df['label'], 'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1]})

# Crear un gráfico de dispersión de 2D con colores basados en las etiquetas de clase
plt.figure(figsize=(12, 8))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="label", palette=palette, data=tsne_df, legend="full", alpha=0.8)
plt.title("t-SNE Visualization")
plt.show()