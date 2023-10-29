# Importamos librerias
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp


# Realizamos la Videocaptura
cap = cv2.VideoCapture(0)
# Mostramos el video en RT


def gen_frame():
    # Empezamos
    while True:
        # Leemos la VideoCaptura
        ret, frame = cap.read()

        # Si tenemos un error
        if not ret:
            break

        else:
            # Corrección de color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Codificamos nuestro video en Bytes
            success, encoded_frame = cv2.imencode('.jpg', frameRGB)
            if not success:
                continue  # Evitar enviar marcos codificados incorrectos

            frame_bytes = encoded_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# Creamos la app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Ruta de aplicacion 'principal'


@app.route('/')
def index():
    return render_template('Index.html')

# Ruta del video


@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ejecutamos la app
if __name__ == "__main__":
    app.run(debug=True)
