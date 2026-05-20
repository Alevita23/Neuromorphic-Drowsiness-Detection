import csi
import image
import time
import ml
import uos
import gc
from ulab import numpy as np

# Crea immagine vuota 320x320 in scala di grigi
# Verrà usata per disegnare l'istogramma degli eventi
img = image.Image(320, 320, image.GRAYSCALE)


# Crea matrice per memorizzare fino a 2048 eventi
# Ogni evento ha 6 valori uint16
events = np.zeros((2048,6), dtype=np.uint16)


# Inizializza la telecamera GENX320
csi0 = csi.CSI(cid=csi.GENX320)

# Reset hardware del sensore
csi0.reset()


# Configura il sensore
csi0.ioctl(

    # Imposta parametri GENX320
    csi.IOCTL_GENX320_SET_MODE,

    # Modalità EVENT:
    csi.GENX320_MODE_EVENT,

    # Numero massimo eventi acquisibili
    events.shape[0]
)

# Caricamento modello TensorFlow Lite
try:

    net = ml.Model(

        # Nome file modello
        "trained.tflite",
        load_to_fb=uos.stat("trained.tflite")[6] >
        (gc.mem_free() - (64*1024))
    )
except Exception as e:

    print(e)

    raise Exception(
        'Errore caricamento trained.tflite'
    )

try:

    labels = [

        line.rstrip("\n")

        for line in open("labels.txt")
    ]

except Exception as e:

    raise Exception(
        'Errore labels.txt'
    )


# Soglia minima di confidenza
# Sotto il 60% classifica come unknown
THRESHOLD = 0.60


# Oggetto per misurare FPS
clock = time.clock()

while True:

    # Aggiorna contatore FPS
    clock.tick()


    # Legge eventi dal sensore
    event_count = csi0.ioctl(

        # comando lettura eventi
        csi.IOCTL_GENX320_READ_EVENTS,

        # salva gli eventi nel buffer
        events
    )


    # Disegna gli eventi nell'immagine
    img.draw_event_histogram(

        # usa solo gli eventi acquisiti
        events[:event_count],

        # pulisce immagine precedente
        clear=True,

        # luminosità pixel
        brightness=128,

        # contrasto immagine
        contrast=64
    )


    # Esegue inferenza del modello ML
    out = net.predict([img])[0]


    try:

        # converte output in lista numerica
        scores = out.flatten().tolist()

    except:

        # se già lista la usa direttamente
        scores = out


    # Cerca indice valore più alto
    best_idx = scores.index(max(scores))

    # probabilità più alta
    best_score = scores[best_idx]


    # Se supera soglia
    if best_score > THRESHOLD:

        # usa etichetta corrispondente
        prediction = labels[best_idx]

    else:

        # altrimenti sconosciuto
        prediction = "unknown"


    # Stampa risultato e FPS
    print(
        prediction,
        best_score,
        "FPS:",
        clock.fps()
    )


    # Aggiorna immagine sul display/framebuffer
    img.flush()
