from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

dibujo = mp.solutions.drawing_utils
direccion = r" " #ruta conjunto de validación
dire_img = os.listdir(direccion)

cap = cv2.VideoCapture(0)

clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

modelo = r" "  #ruta modelo
peso = r" " #ruta peso
cnn = load_model(modelo)
cnn.load_weights(peso)

@app.route('/')
def index():
    return render_template('index.html')

def predict():
    while True:
        ret, frame = cap.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia = frame.copy()
        resultado = manos.process(color)
        posiciones = []
        if resultado.multi_hand_landmarks:
            for mano in resultado.multi_hand_landmarks:
                for id, lm in enumerate(mano.landmark):
                    alto, ancho, c = frame.shape
                    corx, cory = int(lm.x * ancho), int(lm.y * alto)
                    posiciones.append([id, corx, cory])
                    dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                pto_i1 = posiciones[3]
                pto_i2 = posiciones[17]
                pto_i3 = posiciones[10]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1] - 100), (pto_i5[2] - 100)
                ancho, alto = (x1 - 50), (y1 + 100)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)
                if respuesta == 0:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 300, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 300, 0), 1, cv2.LINE_AA)
                elif respuesta == 1:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 300), 3)
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 300), 1, cv2.LINE_AA)
                elif respuesta == 2:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 300), 3)
                    cv2.putText(frame, '{}'.format(dire_img[2]), (x1, y1 - 5), 1, 1.3, (0, 0, 300), 1, cv2.LINE_AA)
                elif respuesta == 3:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 300), 3)
                    cv2.putText(frame, '{}'.format(dire_img[3]), (x1, y1 - 5), 1, 1.3, (0, 0, 300), 1, cv2.LINE_AA)
                elif respuesta == 4:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 300), 3)
                    cv2.putText(frame, '{}'.format(dire_img[4]), (x1, y1 - 5), 1, 1.3, (0, 0, 300), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'LETRA NO IDENTIFICADA', (x1, y1 - 5), 1, 1.3, (0, 0, 300), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(predict(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
