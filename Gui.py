from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import imutils

cap = None

def visualizar():
    global cap, lblVideo
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=640)
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            lblVideo.config(image=photo)
            lblVideo.image = photo
            lblVideo.after(10, visualizar)

def iniciar():
    global cap
    cap = cv2.VideoCapture(0)
    print("Inicio de cámara")
    visualizar()  # Llama a la función visualizar después de iniciar la cámara

def finalizar():
    global cap, pantalla
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    pantalla.destroy()  # Cierra la ventana principal
    print("Fin")

def leerf():
    global modelo, model, det
    modelo = filedialog.askopenfilename(filetypes=[("all modal format", ".pt")])
    if modelo:
        texto4.configure(text=modelo)
    else:
        texto4.configure(text=modelo)

# Ventana principal
# Pantalla
pantalla = Tk()
pantalla.title("Speak Vowel")
pantalla.geometry("1280x720")  # Asignamos la dimension de la ventana

# Fondo
imagenF = PhotoImage(file="Fondo.png")
background = Label(image=imagenF, text="Fondo")
background.place(x=0, y=0, relwidth=1, relheight=1)

# Interfaz
texto1 = Label(pantalla, text="")
texto1.place(x=580, y=10)

texto2 = Label(pantalla, text="")
texto2.place(x=1010, y=100)

texto3 = Label(pantalla, text="Deteccion de modelo: ")
texto3.place(x=110, y=100)

texto4 = Label(pantalla, text="Aun no se ha seleccionado el nombre ")
texto4.place(x=60, y=120)

texto5 = Label(pantalla, text="")
texto5.place(x=110, y=145)

# Botones

# Botones
# Iniciar Video
imagenBI = PhotoImage(file="Inicio.png")
inicio = Button(pantalla, text="Iniciar", image=imagenBI,
                height="40", width="200", command=iniciar)
inicio.grid(row=2, column=0, pady=10)

# Leer
imageBL = PhotoImage(file="rgb.png")
leer = Button(pantalla, text="Leer", image=imageBL,
              height="40", width="200", command=leerf)
leer.grid(row=2, column=1, pady=10)

# Finalizar Video
imagenBF = PhotoImage(file="Finalizar.png")
fin = Button(pantalla, text="Finalizar", image=imagenBF,
             height="40", width="200", command=finalizar)
fin.grid(row=2, column=2, pady=10)



# Video
lblVideo = Label(pantalla)
lblVideo.place(x=320, y=50)


pantalla.mainloop()

