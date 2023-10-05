from sklearn.manifold import TSNE
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

""" t-SNE """
df = pd.read_csv('Crop_recommendation.csv')

# Utiliza seaborn para generar una paleta de colores única basada en los valores únicos en la columna 'Datos'
palette = sns.color_palette('husl', n_colors=len(df['label'].unique()))

# Crea un diccionario que asigne valores únicos a colores de la paleta
color_dict = dict(zip(df['label'].unique(), palette))

# Aplica el mapeo de colores a la columna 'Datos' para crear la nueva columna 'Colores'
df['Colores'] = df['label'].map(color_dict)

df['label']=df['label'].astype('category')


#info df
#df.info()
#df.describe()
#df.isnull().sum()

xWheat = df.drop(['label'], axis=1).drop(['Colores'], axis=1)
yWheat = df['label']
xWheat_scaled = StandardScaler().fit_transform(xWheat)

xWS_df=pd.DataFrame(xWheat_scaled)
xWS_df.columns=['N','P','K','temperature','humidity','ph','rainfall']
xWS_df['label']=yWheat
xWS_df['Colores']=df['Colores']

# xWheat = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# yWheat = df['label']

#xWheat_scaled = sk.preprocessing.StandardScaler().fit_transform(xWheat)

tsneWheat = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsneWheatResults = tsneWheat.fit_transform(xWheat_scaled)

subsetWheatTSNE = pd.DataFrame(yWheat)
subsetWheatTSNE['tsne-2d-one'] = tsneWheatResults[:,0]
subsetWheatTSNE['tsne-2d-two'] = tsneWheatResults[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two", hue="label",
    palette=sns.color_palette("hls", 22), 
    data=subsetWheatTSNE,
    legend="full", alpha=0.8)

plt.show()