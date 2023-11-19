import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from Funciones.condicionales import condicionalesLetras
from Funciones.normalizacionCords import obtenerAngulos

lectura_actual = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

class HandRecognitionApp:
    def __init__(self, root, cap):
        self.root = root
        self.root.title("Reconocimiento de Mano")

        self.cap = cap
        self.width, self.height = 1280, 720
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()

        self.start_button = tk.Button(self.root, text="Iniciar Reconocimiento", command=self.iniciar_reconocimiento)
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="Detener Reconocimiento", command=self.detener_reconocimiento, state=tk.DISABLED)
        self.stop_button.pack()

        self.frame = None
        self.reconocimiento_activo = False

    def iniciar_reconocimiento(self):
        self.reconocimiento_activo = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.actualizar_frame()

    def detener_reconocimiento(self):
        self.reconocimiento_activo = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.root.destroy()  # Cerrar la ventana al detener el reconocimiento

    def actualizar_frame(self):
        global lectura_actual
        if self.reconocimiento_activo:
            ret, frame = self.cap.read()
            if ret:
                height, width, _ = frame.shape
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks is not None:
                    angulosid = obtenerAngulos(results, width, height)[0]
                    dedos = []
                    if angulosid[5] > 125:
                        dedos.append(1)
                    else:
                        dedos.append(0)

                    if angulosid[4] > 150:
                        dedos.append(1)
                    else:
                        dedos.append(0)

                    for id in range(0, 4):
                        if angulosid[id] > 90:
                            dedos.append(1)
                        else:
                            dedos.append(0)

                    TotalDedos = dedos.count(1)
                    condicionalesLetras(dedos, frame)

                    pinky = obtenerAngulos(results, width, height)[1]
                    pinkY = pinky[1] + pinky[0]
                    resta = pinkY - lectura_actual
                    lectura_actual = pinkY

                    if dedos == [0, 0, 1, 0, 0, 0]:
                        if abs(resta) > 30:
                            print("¡Jota en movimiento!")

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                # Convertir la imagen de cv2 a PhotoImage usando Pillow
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(image=img)

                # Mostrar el frame en el widget Canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.photo = photo

                self.root.after(10, self.actualizar_frame)  # Actualizar cada 10 milisegundos

    def run(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75)

        self.root.mainloop()

        self.cap.release()
        cv2.destroyAllWindows()

# Crear la ventana principal
root = tk.Tk()

# Configurar la cámara
cap = cv2.VideoCapture(0)

# Crear la aplicación
app = HandRecognitionApp(root, cap)
app.run()





