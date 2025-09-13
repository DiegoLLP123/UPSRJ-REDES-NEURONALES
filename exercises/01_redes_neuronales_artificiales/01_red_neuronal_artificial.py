import numpy as np
from pprint import pprint as pp
from utils.logger import set_logging, plog
from logging import DEBUG, INFO, WARNING, ERROR

# Llamada a la funcion para configurar el logging
set_logging(log_file='01_red_neuronal_artificial.log')

# Inicializacion de los pesos (w) y el bias (b)
#
# NOTE: Creamos un arreglo de distribucion uniforme de muestras → https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
#       Redondeamos un arreglo al numero de decimales definidos → https://numpy.org/doc/stable/reference/generated/numpy.around.html
#              
weights = np.around(np.random.uniform(size=6), decimals=2)  # pesos sinapticos
biases  = np.around(np.random.uniform(size=3), decimals=2)  # bias

# Impresion de los pesos y bias inicializados
plog(f"Pesos inicializados (w): {weights}")
plog(f"Bias inicializados (b): {biases}", eol=True)

# Definicion de entradas x1, x2
# 
# NOTE: En este ejemplo, consideramos dos entradas (x1, x2)
#       Las entradas pueden ser cualquier valor numerico, en este caso usamos valores entre 0 y 1
#
x1 = 0.5
x2 = 0.85

# Impresion de las entradas
plog(f"Entradas: x1 = {np.around(x1, decimals=4)}, x2 = {np.around(x2, decimals=4)}", eol=True)

# Empecemos calculando la suma ponderada (z11), la cual es la entrada a la neurona 1 en la capa oculta
#
# NOTE: La suma ponderada se calcula como z = x1*w1 + x2*w2 + b
#       donde x1, x2 son las entradas, w1, w2 son los pesos sinapticos asociados a cada entrada y b es el bias
# 
z11 = x1 * weights[0] + x2 * weights[1] + biases[0]

# Impresion de la suma ponderada z11
plog(f"Suma ponderada z11 (neurona 1, capa oculta): {z11 if z11 is None else np.around(z11, decimals=4)}", level=DEBUG, eol=True)

# Ahora, calculamos la suma ponderada (z12), la cual es la entrada a la neurona 2 en la capa oculta
#
# TODO: Completa el calculo de z12
#
z12 = None

# Impresion de la suma ponderada z12
plog(f"Suma ponderada z12 (neurona 2, capa oculta): {z12 if z12 is None else np.around(z12, decimals=4)}", level=ERROR if z12 is None else DEBUG, eol=True)

# Ahora, asumamos que la funcion de activacion es la funcion sigmoide, calculamos la salida a partir de la suma ponderada (z11)
#
# NOTE: La funcion sigmoide se define como σ(z) = 1 / (1 + exp(-z))
#       donde exp es la funcion exponencial
#
a11 = 1.0 / (1.0 + np.exp(-z11))

# Impresion de la salida a11
plog(f"Salida a11 (neurona 1, capa oculta): {a11 if a11 is None else np.around(a11, decimals=4)}", level=DEBUG, eol=True)

# Ahora, calculamos la salida a12 a partir de la suma ponderada (z12)
#
# NOTE: La funcion sigmoide se define como σ(z) = 1 / (1 + exp(-z))
#       donde exp es la funcion exponencial
#
# TODO: Completa el calculo de a12
# 
a12 = None

# Impresion de la salida a12
plog(f"Salida a12 (neurona 2, capa oculta): {a12 if a12 is None else np.around(a12, decimals=4)}", level=ERROR if a12 is None else DEBUG, eol=True)

# Con esto, estas funciones de activacion serviran como entradas para la capa de salida.
# 
# TODO: Calcula la suma ponderada de estas entradas para la neurona de salida. Asigna esta suma a (z2)
#
z2 = None

# Impresion de la suma ponderada z2
plog(f"Suma ponderada z2 (neurona salida): {z2 if z2 is None else np.around(z2, decimals=4)}", level=ERROR if z2 is None else DEBUG, eol=True)

# Finalmente, calculamos la salida final de la red neuronal a partir de la suma ponderada (z2)
#
# NOTE: La funcion sigmoide se define como σ(z) = 1 / (1 + exp(-z))
#       donde exp es la funcion exponencial
#
# TODO: Completa el calculo de a2
#
a2 = None

# Impresion de la salida a2
plog(f"Salida a2 (neurona salida): {a2 if a2 is None else np.around(a2, decimals=4)}", level=ERROR if a2 is None else DEBUG, eol=True)

# Este ejercicio se basó en el Jupyter Notebook creado por Alex Aklson (https://www.linkedin.com/in/aklson/) para 
# © IBM Corporation.