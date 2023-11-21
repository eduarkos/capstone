#------------------------------- Importamos librerias ---------------------------------
#import cv2
import os
import datetime
#---------------------------------Importamos las fotos tomadas-----------------------------
import tensorflow
from tensorflow import keras
#------------------------------ Crear modelo y entrenarlo ---------------------------------------

from tensorflow.python.keras.optimizers import adam_v2  #Optimizador con el que vamos a entrenar el modelo
from tensorflow.python.keras.models import Sequential  #Nos permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D  #Capas para hacer las convoluciones
from tensorflow.python.keras import backend as K       #Si hay una sesion de keras, lo cerramos para tener todo limpio

K.clear_session()  #Limpiamos todo

datos = r'\dataset'

#Parametros
iteraciones = 5 #Numero de iteraciones para ajustar nuestro modelo
altura, longitud = 200, 200 #Tamaño de las imagenes de entrenamiento
batch_size = 1  #Numero de imagenes que vamos a enviar
pasos = 300/1  #Numero de veces que se va a procesar la informacion en cada iteracion
pasos_validacion = 300/1 #Despues de cada iteracion, validamos lo anterior
filtrosconv1 = 32
filtrosconv2 = 64  
filtrosconv3 = 128     #Numero de filtros que vamos a aplicar en cada convolucion
tam_filtro1 = (4,4)
tam_filtro2 = (3,3)
tam_filtro3 = (2,2)   #Tamaños de los filtros 1, 2 y 3
tam_pool = (2,2)  #Tamaño del filtro en max pooling
clases = 29  #Las cinco vocales
lr = 0.0005  #ajustes de la red neuronal para acercarse a una solucion optima

#Pre-Procesamiento de las imagenes
preprocesamiento_entre = keras.utils.image_dataset_from_directory(
    datos,
    labels="inferred",  # Automatically infer class labels from the directory structure
    label_mode="categorical",  # Use one-hot encoded labels
    color_mode="rgb",  # Use color images
    batch_size=32,  # Set the batch size
    image_size=(200, 200),  # Resize images to the desired size
    shuffle=True,  # Shuffle the dataset
    seed=123,  # Set a seed for reproducibility
    validation_split=0.1,  # Split 20% of the data for validation
    subset="training"  # Specify whether to create a training or validation dataset
)

preprocesamiento_vali = keras.utils.image_dataset_from_directory(
    datos,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(200, 200),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"  # Specify that this is a validation dataset
)


#Creamos la red neuronal convolucional (CNN)
cnn = Sequential()  #Red neuronal secuencial
#Agregamos filtros con el fin de volver nuestra imagen muy profunda pero pequeña
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding = 'valid', input_shape=(altura,longitud,3), activation = 'relu')) #Agregamos la primera capa
         #Es una convolucion y realizamos config
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Despues de la primera capa vamos a tener una capa de max pooling y asignamos el tamaño

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'valid', activation='relu')) #Agregamos capa 2
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosconv3, tam_filtro3, padding = 'valid', activation='relu')) #Agregamos capa 3
cnn.add(MaxPooling2D(pool_size=tam_pool))


#Ahora vamos a convertir esa imagen profunda a una plana, para tener 1 dimension con toda la info
cnn.add(Flatten())  #Aplanamos la imagen
cnn.add(Dense(640, activation='relu'))  #Asignamos neuronas
cnn.add(Dropout(0.5)) #Apagamos el 50% de las neuronas en la funcion anterior para no sobreajustar la red
cnn.add(Dense(clases, activation='softmax'))  #Es nuestra ultima capa, es la que nos dice la probabilidad de que sea alguna de las clases

#Agregamos parametros para optimizar el modelo
#Durante el entrenamiento tenga una autoevalucion, que se optimice con Adam, y la metrica sera accuracy

cnn.compile(loss = 'categorical_crossentropy', optimizer= adam_v2.Adam(lr=lr), metrics=['accuracy'])

#Entrenaremos nuestra red
cnn.fit(preprocesamiento_entre, steps_per_epoch=pasos, epochs= iteraciones, validation_data= preprocesamiento_vali, validation_steps=pasos_validacion)

#Guardamos el modelo
cnn.save('ModeloAdamDrop03.h5')
cnn.save_weights('pesos.h5')