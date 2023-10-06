import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap

# Cargar el dataset
df = pd.read_csv('Crop_recommendation.csv')

# Convierte la columna 'label' en una columna categórica
df['label'] = df['label'].astype('category')

# Selecciona las características numéricas
xWheat = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Escala las características
xWheat_scaled = StandardScaler().fit_transform(xWheat)

# Realiza la reducción de dimensionalidad utilizando Isomap en 3D
isomapWheat = Isomap(n_neighbors=35, n_components=3)
manifold_3D = isomapWheat.fit_transform(xWheat_scaled)

# Crea un DataFrame con las componentes principales y la columna 'label'
df_3D = pd.DataFrame(manifold_3D, columns=['Component 1', 'Component 2', 'Component 3'])
df_3D['label'] = df['label']

# Crea una visualización 3D utilizando Plotly Express
fig = px.scatter_3d(df_3D, x='Component 1', y='Component 2', z='Component 3', color='label',
                    title='3D Isomap Graph')
fig.show()