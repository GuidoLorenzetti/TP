import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sklearn as sk

df = pd.read_csv('Crop_recommendation.csv')
# Utiliza seaborn para generar una paleta de colores única basada en los valores únicos en la columna 'Datos'
palette = sns.color_palette('husl', n_colors=len(df['label'].unique()))

# Crea un diccionario que asigne valores únicos a colores de la paleta
color_dict = dict(zip(df['label'].unique(), palette))

# Aplica el mapeo de colores a la columna 'Datos' para crear la nueva columna 'Colores'
df['colors'] = df['label'].map(color_dict)

df['label']=df['label'].astype('category')

# Creamos una copia del dataframe en un nuevo csv
df.to_csv('Crop_recommendation_corrected.csv', index=False)