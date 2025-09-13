import numpy as np
from utils.logger import set_logging, plog
from logging import DEBUG, INFO, WARNING, ERROR
from pprint import pformat

# Llamada a la funcion para configurar el logging
set_logging(log_file='02_red_pequena.log')

# TODO: Crea una funcion 'initialize_network' que inicialice una red neuronal pequeña
#       con un numero definido de entradas, capas ocultas, neuronas por capa y salidas.
#       La funcion debe retornar un diccionario representando la red neuronal, donde cada capa contiene 
#       nodos con sus pesos y bias inicializados aleatoriamente.
def initialize_network(num_inputs, num_hidden_layers, neurons_per_hidden, num_outputs):
    return None  # Reemplaza con tu codigo

# La suma ponderada en cada neurona es calculada como el producto punto de las 
# entradas y los pesos mas los biases: z = Σ(x_i * w_i) + b
# Creemos una funcion para calcular la suma ponderada
def get_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

# Recordemos que la salida de cada neurona es simplemente una transformacion no lineal de la suma ponderada.
# Usamos las funciones de activacion para esto.
# Implementemos la funcion sigmoide como ejemplo de funcion de activacion
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

# La pieza final de la construccion de una red neuronal que puede hacer predicciones es poner todo junto.
# Implementemos la funcion 'forward_propagate' que toma una red neuronal y un conjunto de entradas,
# y propaga las entradas a traves de la red para obtener las salidas.
#
# NOTE: La manera en que esto funciona es:
#   1. Inicialmente, las entradas son pasadas a la primera capa oculta.
#   2. Calculamos la suma ponderada en las neuronas de la capa actual usando los pesos y bias de la neurona.
#   3. Calculamos la salida de las neuronas de la capa actual usando la funcion de activacion.
#   4. Las salidas de la capa actual se convierten en las entradas para la siguiente capa.
#   5. Nos movemos a la siguiente capa en la red.
#   6. Repetimos los pasos 2-5 para todas las capas ocultas hasta llegar a la capa de salida.
# 
# TODO: Completa la funcion 'forward_propagate' para que implemente el proceso descrito arriba.
#       La funcion debe retornar las predicciones de la red neuronal.
def forward_propagate(network, inputs):
    return None  # Reemplaza con tu codigo

if __name__ == "__main__":
    
    # Inicializamos una red neuronal pequeña
    num_inputs = 2
    num_hidden_layers = 1
    neurons_per_hidden = 2
    num_outputs = 1

    network = initialize_network(num_inputs, num_hidden_layers, neurons_per_hidden, num_outputs)
    
    # Impresion de la red neuronal inicializada
    plog(f"Red neuronal:\n{pformat(network, indent=4)}", level=ERROR if network is None else DEBUG, eol=True)

    # Generamos 5 entradas de ejemplo con las cuales probaremos la red neuronal
    np.random.seed(12)
    inputs = np.around(np.random.uniform(size=(5, num_inputs)), decimals=2)

    # Impresion de las entradas
    plog(f"Entradas: x1 = {inputs}, x2 = {inputs}", eol=True)
    
    # TODO: Usa la funcion 'weighted_sum' para calcular la suma ponderada en la neurona 1 de la capa oculta
    #
    weights = None  # Reemplaza con los pesos de la neurona 1 en la capa oculta
    bias    = None  # Reemplaza con el bias de la neurona 1 en la capa oculta
    
    weighted_sum = None  # Reemplaza con el calculo de la suma ponderada usando la funcion 'get_weighted_sum'
    
    # Impresion de la suma ponderada
    plog(f"Suma ponderada (neurona 1, capa oculta): {weighted_sum if weighted_sum is None else np.around(weighted_sum, decimals=4)}", level=ERROR if weighted_sum is None else DEBUG, eol=True)
    
    # TODO: Usa la funcion 'node_activation' para calcular la salida de la neurona 1 en la capa oculta
    #
    node_output = None  # Reemplaza con el calculo de la salida usando la funcion 'node_activation'
    
    # Impresion de la salida
    plog(f"Salida (neurona 1, capa oculta): {node_output if node_output is None else np.around(node_output, decimals=4)}", level=ERROR if node_output is None else DEBUG, eol=True)

    # TODO: Usa la funcion 'forward_propagate' para propagar las entradas a traves de la red neuronal
    #
    predictions = None  # Reemplaza con el calculo de las predicciones usando la funcion 'forward_propagate'
    
    # Impresion de las predicciones
    plog(f"Predicciones: {predictions if predictions is None else np.around(predictions, decimals=4)}", level=ERROR if predictions is None else DEBUG, eol=True)

# Este ejercicio se basó en el Jupyter Notebook creado por Alex Aklson (https://www.linkedin.com/in/aklson/) para 
# © IBM Corporation.