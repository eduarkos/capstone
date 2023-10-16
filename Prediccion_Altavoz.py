import cv2
import mediapipe as mp
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pyttsx3

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

modelo = r'C:\Users\56998\Downloads\FILE_HAND_SIGHT-20230827T231047Z-001\FILE_HAND_SIGHT\Deteccion-y-Clasificacion-de-Manos-main\Modelo.h5'
peso =  r'C:\Users\56998\Downloads\FILE_HAND_SIGHT-20230827T231047Z-001\FILE_HAND_SIGHT\Deteccion-y-Clasificacion-de-Manos-main\pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(peso)

direccion = r'C:\Users\56998\Desktop\fotos_vocales\Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

cap = cv2.VideoCapture(0)

clase_manos  =  mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

# Estado anterior de la letra detectada
letra_anterior = None

while (1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto)
                posiciones.append([id,corx,cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                pto_i1 = posiciones[3]
                pto_i2 = posiciones[17]
                pto_i3 = posiciones[10]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1,y1 = (pto_i5[1]-100),(pto_i5[2]-100)
                ancho, alto = (x1-80),(y1+160)
                x2,y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)
                
                if respuesta in range(len(dire_img)):
                    letra_identificada = dire_img[respuesta]
                    
                    # Verificar si la letra detectada es diferente de la letra anterior
                    if letra_identificada != letra_anterior:
                        print(f"Letra identificada: {letra_identificada}")
                        
                        # Reproducir el nombre por altavoz
                        engine.say(letra_identificada)
                        engine.runAndWait()

                        letra_anterior = letra_identificada  # Actualizar el estado de la letra anterior
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 300, 0), 3)
                    cv2.putText(frame, '{}'.format(letra_identificada), (x1, y1 - 5), 1, 1.3, (0, 300, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'LETRA NO IDENTIFICADA', (x1, y1 - 5), 1, 1.3, (0, 0, 300), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
