import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap


df = pd.read_csv("Crop_recommendation.csv")


# Utiliza seaborn para generar una paleta de colores única basada en los valores únicos en la columna 'Datos'
palette = sns.color_palette('husl', n_colors=len(df['label'].unique()))

# Crea un diccionario que asigne valores únicos a colores de la paleta
color_dict = dict(zip(df['label'].unique(), palette))

# Aplica el mapeo de colores a la columna 'Datos' para crear la nueva columna 'Colores'
df['Colores'] = df['label'].map(color_dict)

#info df
#df.info()
#df.describe()
#df.isnull().sum()

df['label']=df['label'].astype('category')

xWines = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
yWines = df['label']

xWines_scaled = sk.preprocessing.StandardScaler().fit_transform(xWines)

####2D########
# isomapWheat = Isomap(n_neighbors=6, n_components=2)
# isomapWheat.fit(xWines_scaled)
# manifold_2Da = isomapWheat.transform(xWines_scaled)
# manifold_2D = pd.DataFrame(manifold_2Da,
#     columns=['Component 1', 'Component 2'])
# manifold_2D['category'] = yWines.to_numpy()

# groups = manifold_2D.groupby('category')
# plt.title('2D Isomap Graph')
# for name, group in groups:
#     plt.plot(group['Component 1'], group['Component 2'], marker='o', linestyle='', markersize=5, label=name)
# plt.legend()


####3D########
isomapWheat = Isomap(n_neighbors=6, n_components=3)
isomapWheat.fit(xWines_scaled)
manifold_3Da = isomapWheat.transform(xWines_scaled)
manifold_3D = pd.DataFrame(manifold_3Da,
    columns=['Component 1', 'Component 2','Component 3'])
manifold_3D['category'] = yWines.to_numpy()
fig = px.scatter_3d(manifold_3D, x='Component 1', y='Component 2', z='Component 3', color='category',
                    title='3D Isomap Graph')
fig.show()
