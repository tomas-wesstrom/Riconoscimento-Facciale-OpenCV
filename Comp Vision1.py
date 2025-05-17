# Importa i moduli necessari
import cv2 as cv
import time
import os
import numpy as np
import shutil

# Directory per salvare le immagini delle facce rilevate dalla webcam
FACES_DIR = "detected_faces"
# Directory contenente le immagini utilizzate per l'addestramento del riconoscimento facciale
TRAINING_DIR = "training_images"

# Soglia di tolleranza per confrontare le codifiche delle facce.
FRAMES_TO_CONFIRM = 10

# Dizionari per il conteggio e lo stato.
detection_counts = {}
last_detection_state = {}
recognition_counts = {}
last_recognition = {}
face_photo_taken = {}

# Percorso al file Haar Cascade
CASCADE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'haarcascades', 'haarcascade_frontalface_default.xml')
)
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Haar Cascade file not found: {CASCADE_PATH}")
face_cascade = cv.CascadeClassifier(CASCADE_PATH)

# --- TRAINING LBPH MODEL IF NOT EXISTS ---
LBPH_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trainedData.yml')
TRAINING_DIR = os.path.join(os.path.dirname(__file__), 'training_images')

# Build name_dict dynamically from training_images subfolders
name_dict = {}
if os.path.exists(TRAINING_DIR):
    for idx, folder in enumerate(sorted(os.listdir(TRAINING_DIR)), 1):
        if os.path.isdir(os.path.join(TRAINING_DIR, folder)):
            name_dict[idx] = folder
    print(f"Dynamic name_dict: {name_dict}")

if not os.path.exists(LBPH_MODEL_PATH):
    print("Modello LBPH non trovato, avvio training...")
    faces = []
    labels = []
    for idx, folder in name_dict.items():
        folder_path = os.path.join(TRAINING_DIR, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv.resize(img, (200, 200)))
                    labels.append(idx)
    if not faces:
        print("Nessuna immagine trovata per l'addestramento.")
        exit(1)
    labels = np.array(labels)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.save(LBPH_MODEL_PATH)
    print(f"Modello LBPH addestrato e salvato in {LBPH_MODEL_PATH}")
else:
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(LBPH_MODEL_PATH)
    print(f"Modello LBPH caricato da {LBPH_MODEL_PATH}")
lbph_enabled = True

# Pulisce la directory delle facce rilevate.
if os.path.exists(FACES_DIR):
    try:
        shutil.rmtree(FACES_DIR)
        os.makedirs(FACES_DIR)
        print(f"Pulita e ricreata la directory: {FACES_DIR}")
    except OSError as e:
        print(f"Errore nella pulizia della directory {FACES_DIR}: {e}")
else:
    os.makedirs(FACES_DIR)
    print(f"Creata la directory: {FACES_DIR}")

# Apri il flusso video dalla webcam.
webcam_video_stream = cv.VideoCapture(0)
if not webcam_video_stream.isOpened():
    print("Errore: Impossibile accedere alla camera.")
    exit()
else:
    print("Camera accessibile.")

frame_count = 0
process_every_n_frames = 3

photo_taken = False

while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Fallito il recupero del frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    face_names_with_confidence = []
    for (x, y, w, h) in faces:
        label = "???"
        confidence = 0.0
        if lbph_enabled:
            region = gray[y:y+h, x:x+w]
            try:
                id_pred, conf = recognizer.predict(region)
                if conf < 100:
                    label = name_dict.get(id_pred, str(id_pred))
                    confidence = conf
            except Exception as e:
                print(f"LBPH prediction error: {e}")
        face_names_with_confidence.append(((y, x+w, y+h, x), (label, confidence)))

    for i, ((top, right, bottom, left), (name, confidence)) in enumerate(face_names_with_confidence):
        face_id = i
        face_image = frame[top:bottom, left:right]
        detection_state = name if name != "???" else "unknown"

        # Scatta una sola foto per sessione, solo se non già fatta
        if not photo_taken:
            filename_detected = os.path.join(FACES_DIR, f"{detection_state}_{int(time.time())}.jpg")
            cv.imwrite(filename_detected, face_image)
            print(f"Foto scattata e salvata come: {filename_detected}")
            photo_taken = True

        label = f"{name} ({confidence:.2f})" if name != "???" else name
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        font_thickness = 1
        text_size = cv.getTextSize(label, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
        rect_width = right - left
        while text_width > rect_width - 10:
            font_scale *= 0.9
            text_size = cv.getTextSize(label, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size
            if font_scale < 0.5:
                break
        text_x = left + 6
        text_y = bottom - 6
        cv.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    cv.imshow("Rilevamento Facciale", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv.destroyAllWindows()
