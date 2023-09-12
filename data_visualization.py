import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Crop_recommendation_corrected.csv')

corr = df.drop(['label'], axis=1).corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    annot_kws = {'size': 6}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

sns.set(style="darkgrid")

# Obtiene una lista de todas las columnas numéricas del DataFrame
numeric_columns = df.select_dtypes(include=[float, int]).columns

# Calcula el número de filas y columnas necesarias para el ploteo
num_rows = (len(numeric_columns) + 1) // 2
num_cols = 2

fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))

# Colores distintos para los histogramas
colors = ["skyblue", "salmon", "olive", "gold", "teal", "purple"]

# Itera a través de las columnas numéricas y crea un histograma para cada una
for i, col in enumerate(numeric_columns):
    row = i // num_cols
    col_idx = i % num_cols
    
    # Selecciona un color diferente para cada histograma
    color_idx = i % len(colors)
    color = colors[color_idx]
    
    sns.histplot(data=df, x=col, kde=True, color=color, ax=axs[row, col_idx])
    
    # Configura el título debajo del gráfico
    axs[row, col_idx].set_title("")
    axs[row, col_idx].set_xlabel(col)

# Elimina cualquier subplot no utilizado
for i in range(len(numeric_columns), num_rows * num_cols):
    row = i // num_cols
    col_idx = i % num_cols
    fig.delaxes(axs[row, col_idx])

# Añade un título general a la figura
plt.suptitle("Histogramas de Variables Numéricas")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta el espacio para el título general
plt.show()