#---------------------- IMPORTACIÓN DE LIBERIAS --------------------
import numpy as np
import pandas as pd
from warnings import warn
from scipy.interpolate import griddata


def matrix_generator(coords, size: list, offset: list = [0,0],
                     labels: list = ['x', 'y', 'kappa'], keep_kappa = True,
                     return_movement_vector: bool = False,
                     retun_cell_size: bool = False, interpolation_type = 'book'):

  #-------------------------- VERIFICACIONES -------------------------
  '''
  Aquí, las verificaciones son cruciales, porque en caso de no preever un caso
  que pueda generar un bug, se puede perder mucho tiempo en una ejecución no
  exitosa.

  Sería buena idea poner un warning de una resolución mínima/máxima y de un
  offset mínimo y máximo.
  '''

  if not coords.columns.tolist() == labels:
    raise ValueError(
    f'La lista proporcionada de nombre de columnas {labels} '
    f'no coincide con la del dataframe {coords.columns.tolist()}')

  # Verificaciones de longitud de size y offset
  if len(size) != 2:
    raise ValueError('size debe tener 2 elementos')
  if len(offset) != 2:
    raise ValueError('offset debe tener 2 elementos')

  # Verificaciones no letales de los valores de size y offset
  offset_flag = True

  for i in range(2):
    if size[i]%1 != 0:
      warn(f'el size brindado ({size}) es no entero,'
            f'se redondeará a {round(size[i])}')
      size[i] = int(round(size[i]))
    else:
      size[i] = int(size[i])

    """
    Si offset es una cadena con número, se va a tomar como porcentaje
    """
    if isinstance(offset[i], str):
      try:
        val = float(offset[i])
        offset[i] = val * size[i] / 100
      except ValueError:
        raise ValueError(f'offset[{i}] = {offset[i]} no es numérico')


    if offset[i]%1 != 0:
      warn(f'el offset brindado ({offset}: {offset[i]}) es no entero,'
            f'se redondeará a {round(offset[i])}')
      offset[i] = int(round(offset[i]))
    else:
      offset[i] = int(offset[i])

    if size[i] < 0:
      warn(f'el size brindado ({size}: {size[i]})) es negativo.'
            f'se cambiará su signo ({-size})')
      size[i] = -size[i]

    if offset[i] < 0:
      warn(f'el ofset brindado ({offset}: {offset[i]}) es negativo.'
            f'se cambiará su signo ({-offset[i]})')
      offset[i] = -offset[i]

    if offset[i] == 0 and offset_flag:
      offset_flag = False
      warn('Un ofset nulo puede causar errores de interpolación')

  # ------------------------- INICIALIZACION  -------------------------
  #crear la matriz
  matrix = np.zeros((size[0] + 2*offset[0],
                    size[1] + 2*offset[1]))

  # Desempaquetar datos
  kappa = coords[labels[2]].to_numpy()
  # Vector de posición
  r = np.column_stack([coords[labels[0]].to_numpy(), # x
                      coords[labels[1]].to_numpy()]) # y
  """
  El vector de posición almacena los datos de esta manera:
  [número de vértice][x (0) o y (1)]
  """

  # ------------------ AJUSTE DE COORDENADAS A MATRIZ ------------------
  # Calcular límites de espacio
  lim_x = np.array([r[:, 0].min(), r[:, 0].max()])
  lim_y = np.array([r[:, 1].min(), r[:, 1].max()])
  # Encontrar la normalización
  space_range = [lim_x[1]- lim_x[0], lim_y[1]- lim_y[0]]
  normalization_factor = max(space_range)/min(size)
  # Vector de movimiento (todos los puntos son positivos)
  movement_vector  = np.array([lim_x[0], lim_y[0]])

  # Aplicación de las modificaciones de las coordenadas
  r-= movement_vector # Movimiento para hacer positivos (coord fisicas)
  r /= normalization_factor # Encajar las coords en la grilla (coord de grilla)
  r += np.array([offset[0], offset[1]]) # Movimiento de offset (coord grilla)
  # Normalización del vector de movimiento
  movement_vector /= normalization_factor # (coord grilla)
  movement_vector -= np.array([offset[0], offset[1]]) # Movimiento de offset

  # Tamaño de la celda (cuadrangular)
  cell_size = normalization_factor

  # ------------------------- INTERPOLACIÓN  -------------------------

  if interpolation_type == 'book':
    # Interpolación de CIC
    # Coordenadas en grilla
    x = r[:, 0]
    y = r[:, 1]

    # Índices base
    i = np.floor(x).astype(int)
    j = np.floor(y).astype(int)

    # Fracciones dentro de la celda (adimensionales)
    dx = x - i
    dy = y - j

    # Pesos CIC (fracción de área, ya normalizados)
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    # Dimensiones de la matriz
    Ny, Nx = matrix.shape

    # Filtrado de puntos válidos (evitar bordes fuera)
    mask = (i >= 0) & (j >= 0) & (i + 1 < Nx) & (j + 1 < Ny)

    i = i[mask]
    j = j[mask]
    k = kappa[mask]

    w00 = w00[mask]
    w10 = w10[mask]
    w01 = w01[mask]
    w11 = w11[mask]

    # Área de celda
    A = cell_size**2

    # Acumulación vectorizada
    if keep_kappa:
      # κ distribuida (circulación por celda)
      np.add.at(matrix, (j,     i    ), k * w00)
      np.add.at(matrix, (j,     i + 1), k * w10)
      np.add.at(matrix, (j + 1, i    ), k * w01)
      np.add.at(matrix, (j + 1, i + 1), k * w11)
    else:
      # ω física (densidad de vorticidad)
      np.add.at(matrix, (j,     i    ), k * w00 / A)
      np.add.at(matrix, (j,     i + 1), k * w10 / A)
      np.add.at(matrix, (j + 1, i    ), k * w01 / A)
      np.add.at(matrix, (j + 1, i + 1), k * w11 / A)

  if interpolation_type == 'griddata':
    # Coordenadas de los puntos
    points = r  # (N,2)
    values = kappa

    # Grilla
    Ny, Nx = matrix.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(Nx),
        np.arange(Ny)
    )

    # Interpolación
    matrix = griddata(points, values, (grid_x, grid_y), method='linear')
    matrix[np.isnan(matrix)] = 0

  # ------------------------- RETURNS -------------------------
  if return_movement_vector and retun_cell_size:
    return matrix, r, cell_size, movement_vector
  elif return_movement_vector:
    return matrix, r, movement_vector
  elif retun_cell_size:
    return matrix, r, cell_size
  else:
    return matrix, r
