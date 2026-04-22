#---------------------- IMPORTACIÓN DE LIBERIAS --------------------
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def plot_vortex(coords, colors = 'magma',
                labels = ['x', 'y', 'kappa']):
  #-------------------------- VERIFICACIONES -------------------------
  if not coords.columns.tolist() == labels:
    raise ValueError(
    f'La lista proporcionada de nombre de columnas {labels} '
    f'no coincide con la del dataframe {coords.columns.tolist()}')

  #-------------------------- AJUSTE DE COLOR  -------------------------
  # Cálculo de la dispersión de kappa
  vmax = coords[labels[2]].max()
  vmin = coords[labels[2]].min()
  # Obtención del mapa de colores
  cmap = plt.get_cmap(colors, 100)
  if vmax != vmin:
    # Normalización normal si hay variedad de datos
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    assinged_colors = cmap(norm(coords[labels[2]]))
  else:
    assinged_colors = [cmap(0.5)] * len(coords)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



  # ------------------------- GRAFICACION -------------------------
  fig, ax = plt.subplots(figsize = (10, 10))
  ax.scatter(coords[labels[0]], coords[labels[1]],
             c = assinged_colors)

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  fig.colorbar(sm, ax=ax, label=f'Valores de {labels[2]}')

  ax.set_aspect('equal')
  plt.show()