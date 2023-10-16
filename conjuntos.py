import os
import random
import shutil

# Ruta de la carpeta que contiene todas las imágenes
ruta_imagenes = r'C:\Users\56998\Desktop\fotos_vocales\todo'

# Ruta de la carpeta de entrenamiento y validación
ruta_entrenamiento = r'C:\Users\56998\Desktop\fotos_vocales\Entrenamiento'
ruta_validacion = r'C:\Users\56998\Desktop\fotos_vocales\Validacion'

# Proporción de datos para validación (por ejemplo, 20%)
proporcion_validacion = 0.2

# Lista de las clases
clases = os.listdir(ruta_imagenes)

# Crear carpetas para cada clase en los conjuntos de entrenamiento y validación
for clase in clases:
    carpeta_entrenamiento = os.path.join(ruta_entrenamiento, clase)
    carpeta_validacion = os.path.join(ruta_validacion, clase)
    os.makedirs(carpeta_entrenamiento, exist_ok=True)
    os.makedirs(carpeta_validacion, exist_ok=True)

# Iterar sobre cada clase y dividir las imágenes en entrenamiento y validación
for clase in clases:
    carpeta_clase = os.path.join(ruta_imagenes, clase)
    imagenes_clase = os.listdir(carpeta_clase)
    
    # Mezclar aleatoriamente las imágenes de la clase
    random.shuffle(imagenes_clase)
    
    # Calcular el número de imágenes para entrenamiento y validación
    total_imagenes_clase = len(imagenes_clase)
    num_imagenes_validacion = int(proporcion_validacion * total_imagenes_clase)
    num_imagenes_entrenamiento = total_imagenes_clase - num_imagenes_validacion
    
    # Mover las imágenes a las carpetas de entrenamiento y validación de la clase
    for i, imagen in enumerate(imagenes_clase):
        origen = os.path.join(carpeta_clase, imagen)
        if i < num_imagenes_entrenamiento:
            destino = os.path.join(os.path.join(ruta_entrenamiento, clase), imagen)
        else:
            destino = os.path.join(os.path.join(ruta_validacion, clase), imagen)
        shutil.move(origen, destino)

# Ahora tienes las imágenes divididas aleatoriamente en conjuntos de entrenamiento y validación para cada clase

