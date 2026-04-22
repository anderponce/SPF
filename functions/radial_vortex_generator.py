#---------------------- IMPORTACIÓN DE LIBRERIAS --------------------
import numpy as np
import pandas as pd
import os
from warnings import warn

def radial_vortex_generator(R: float, dr: float, n_theta: int,
                            save_loc: str = None, kappa: float = 3.0,
                            center: bool = True, plot: bool = False,
                            labels: list = ['x', 'y', 'kappa'],
                            distrubution = 0, rad_grad: float = 1):

  #---------------------- ENLISTADO DE POSIBLES MODOS --------------------
  # modos de distribución
  constant_names = ('constant', 0, '0', 'c', 'co', 'cte')
  alt_names = ('alt', 'a', 'alternating', -1, '-1')
  radial_names = ('radial', 'r', 'rad', '1', 1)
  alt_rad_names = ('alt rad', 'alt_rad', 'alternating radial','rad alt',
                   'rad_alt', 'ra', 'r_a', 'r a', 'ar' , 'a_r', 'a r',
                   'radial alternating')


  distribution_names = constant_names+ alt_names + radial_names + alt_rad_names

  #-------------------------- VERIFICACIONES -------------------------
  # Revisiones de instancias
  if len(labels) != 3:
    raise ValueError('labels debe tener 3 elementos')

  # Warnings de redondeos
  if R%dr != 0:
    warn( f'El número de radios {round(R/dr,4)} no es un número entero. '
          f'Se redondeará a {round(R/dr)}')

  if distrubution not in distribution_names:
    raise ValueError('Modo de distribución, no disponible.')

  #--------------------- CÁLCULOS INICIALIZADORES --------------------
  # Cálculo del número de radios
  nr = round(R / dr)

  # Número de vórtices \sum_{i=1}^{n_r}[n_theta*1]
  N = int(n_theta * nr*(nr +1 )/2)

  if center:
    N += 1

  # Generación de la matriz vacía de coordenadas (x,y,kappa)
  coords = np.zeros((N, 3))

  if center:
    coords[-1, 2] = kappa


  #-------------------- GENERACIÓN DE LOS VORTICES -------------------
  for i in range(1, nr+1):
    # Calcula el tamaño actual del radio
    r = i * dr

    # Encontramos donde inicia y termina el slice para guardar los datos
    ini = int(n_theta * i * (i - 1) // 2)
    end = ini + (n_theta * i)


    # Creación de los espacios vórtices con ángulos equidistantes
    theta = np.linspace(0, 2*np.pi,
                        n_theta*i,  # Crece el número de vórtices por radios
                        endpoint=False)

    # Calcula y guarda las coordenadas de cada vórtice
    coords[ini:end, 0] += r * np.cos(theta)
    coords[ini:end, 1] += r * np.sin(theta)

    if distrubution in constant_names:
      coords[ini:end, 2] += kappa
    elif distrubution in alt_names:
      coords[ini:end, 2] += kappa * (-1)**i
    elif distrubution in radial_names:
      if rad_grad >= 0:
        coords[ini:end, 2] += kappa * (i)**rad_grad
      else:
        coords[ini:end, 2] += kappa * (i+1)**rad_grad
    elif distrubution in alt_rad_names:
      if rad_grad >= 0:
        coords[ini:end, 2] += kappa * (i) * (-1)**i
      else:
        coords[ini:end, 2] += kappa * (i)**rad_grad * (-1)**i

  #-------------------- ALMACENAMIENTO DE DATOS -------------------
  # Convierte las coordenadas en un data frame
  coords = pd.DataFrame(coords, columns=labels)

  # Guarda el data frame si se coloca una ruta
  if save_loc is not None:
    # Extraemos el directorio para no crear una carpeta con nombre de archivo
    directorio = os.path.dirname(save_loc)
    if directorio == "":
      directorio = "." # Por si el usuario solo pone "datos.csv"

    # Revisa si existe el directorio
    if not os.path.exists(directorio):

      # Contador de intentos. Iniciado en 5 para ejecutar el menú,
      attempts_counter = 5

      # Repetición de menú
      while True:
        # Repetición de menú cada 5 intentos
        if attempts_counter >= 5:
          try:
            r = int(input(f'Warning: No se encontró el directorio {directorio}. '
                    '¿Desea crearlo? \n\t [1]: Si, [0]: No. '))
          except ValueError:
            r = -1 # Evita que explote si el usuario presiona Enter sin querer
          attempts_counter = 0

        # Marcar Opciones
        if r == 1:
          os.makedirs(directorio, exist_ok=True)
          coords.to_csv(save_loc, index=False) # Faltaba guardar aquí
          break
        elif r == 0:
          warn('Continuando sin guardar.')
          break
        else:
          print('Opción no válida. Intente de nuevo')
          attempts_counter += 1
    else:
      # Guarda el dataframe en un csv con la ubicación saveloc
      coords.to_csv(save_loc, index=False)

  # Devuelve el dataframe
  return coords