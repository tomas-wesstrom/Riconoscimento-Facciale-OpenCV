# Importa i moduli necessari
import cv2 as cv  # Libreria OpenCV per l'elaborazione delle immagini
import time  # Modulo per misurare il tempo
import os  # Modulo per interagire con il sistema operativo (es. gestione dei file)
import numpy as np  # Libreria per operazioni numeriche
from PIL import Image  # Python Imaging Library
import shutil  # Modulo per operazioni sui file ad alto livello

# Directory per salvare le immagini delle facce rilevate dalla webcam
FACES_DIR = "detected_faces"
# Directory contenente le immagini utilizzate per l'addestramento del riconoscimento facciale
TRAINING_DIR = "training_images"

# Soglia di confidenza per LBPH (più bassa = più sicuro, tipico 50-80)
LBPH_CONFIDENCE_THRESHOLD = 70

# Numero di frame consecutivi in cui una faccia deve essere rilevata/riconosciuta prima di essere salvata.
FRAMES_TO_CONFIRM = 10

# Dizionari per tenere traccia del conteggio dei rilevamenti/riconoscimenti per ogni volto.
detection_counts = {}
last_detection_state = {}
recognition_counts = {}
last_recognition = {}

# Set per memorizzare i nomi delle facce già salvate per evitare duplicati.
faces_saved_detected = set()
faces_saved_training = set()

# Carica il classificatore Haar Cascade per il rilevamento dei volti
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(CASCADE_PATH)

# File per salvare il modello LBPH
TRAINER_FILE = "trainer.yml"

# Funzione per caricare i dati di training per LBPH
# Ritorna: immagini (grayscale), labels (int), label2name (dict)
def load_training_data_lbph(training_dir):
    images = []
    labels = []
    label2name = {}
    name2label = {}
    current_label = 0
    files_without_faces = []
    for person_name in os.listdir(training_dir):
        person_path = os.path.join(training_dir, person_name)
        if os.path.isdir(person_path):
            if person_name not in name2label:
                name2label[person_name] = current_label
                label2name[current_label] = person_name
                current_label += 1
            label = name2label[person_name]
            for filename in os.listdir(person_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    filepath = os.path.join(person_path, filename)
                    try:
                        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"Immagine non caricata: {filepath}")
                            continue
                        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                face_img = img[y:y+h, x:x+w]
                                face_img = cv.resize(face_img, (200, 200))
                                images.append(face_img)
                                labels.append(label)
                                break  # Solo la prima faccia
                        else:
                            print(f"Nessuna faccia trovata in: {filepath}")
                            files_without_faces.append(filepath)
                    except Exception as e:
                        print(f"Errore nel caricamento o codifica di {filepath}: {e}")
    return images, labels, label2name, files_without_faces

# Carica i dati di training.
train_images, train_labels, label2name, files_to_remove = load_training_data_lbph(TRAINING_DIR)
print(f"Caricate {len(train_images)} immagini di volti noti da {len(label2name)} persone.")

# Rimuove le immagini senza volto dalla directory di training.
for filepath in files_to_remove:
    try:
        os.remove(filepath)
        print(f"Rimossa immagine senza volto: {filepath}")
    except OSError as e:
        print(f"Errore nella rimozione di {filepath}: {e}")

# Addestra o carica il riconoscitore LBPH
recognizer = cv.face.LBPHFaceRecognizer_create()
if os.path.exists(TRAINER_FILE):
    recognizer.read(TRAINER_FILE)
    print(f"Modello LBPH caricato da {TRAINER_FILE}")
else:
    if len(train_images) > 0:
        recognizer.train(train_images, np.array(train_labels))
        recognizer.save(TRAINER_FILE)
        print(f"Modello LBPH addestrato e salvato in {TRAINER_FILE}")
    else:
        recognizer = None
        print("Nessuna immagine di training valida trovata!")

# Pulisce la directory delle facce rilevate all'avvio.
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

# Loop principale per l'elaborazione dei frame video.
while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Fallito il recupero del frame")
        break

    process_frame = True  # Processa sempre il frame
    face_locations = []
    face_names_with_confidence = []

    if process_frame:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in detected:
            face_img = gray[y:y+h, x:x+w]
            face_img_resized = cv.resize(face_img, (200, 200))
            name = "???"
            confidence = 0.0
            if recognizer is not None:
                label_pred, conf = recognizer.predict(face_img_resized)
                if conf < LBPH_CONFIDENCE_THRESHOLD:
                    name = label2name.get(label_pred, "???")
                    confidence = 1.0 - (conf / 100.0)
                else:
                    confidence = 1.0 - (conf / 100.0)
            face_locations.append((y, x + w, y + h, x))
            face_names_with_confidence.append((name, confidence))
        if detected is not None and len(detected) > 0:
            last_face_locations = face_locations
            last_face_names_with_confidence = face_names_with_confidence
        else:
            last_face_locations = []
            last_face_names_with_confidence = []
            detection_counts = {}
            last_detection_state = {}
            recognition_counts = {}
            last_recognition = {}
    else:
        face_locations = last_face_locations
        face_names_with_confidence = last_face_names_with_confidence

    # Mostra i risultati sul frame e gestisce il salvataggio delle facce.
    for i, ((top, right, bottom, left), (name, confidence)) in enumerate(zip(face_locations, face_names_with_confidence)):
        face_id = i
        face_image = frame[top:bottom, left:right]
        detection_state = name if name != "???" else "unknown"

        # Logica per salvare le facce rilevate nella directory 'detected_faces'.
        if face_id not in last_detection_state:
            last_detection_state[face_id] = detection_state
            detection_counts[face_id] = 1
        elif last_detection_state[face_id] == detection_state:
            detection_counts[face_id] += 1
            if detection_counts[face_id] >= FRAMES_TO_CONFIRM:
                detected_face_id = name if name != "???" else f"unknown_{face_id}"
                if detected_face_id not in faces_saved_detected:
                    filename_detected = os.path.join(FACES_DIR, f"{detected_face_id}.jpg")
                    cv.imwrite(filename_detected, face_image)
                    print(f"DETECTED (CONFIRMED): Faccia salvata come: {filename_detected}")
                    faces_saved_detected.add(detected_face_id)
        else:
            last_detection_state[face_id] = detection_state
            detection_counts[face_id] = 1

        # Logica per salvare le facce riconosciute nella cartella di training.
        if name != "???":
            if face_id not in last_recognition:
                last_recognition[face_id] = name
                recognition_counts[face_id] = 1
            elif last_recognition[face_id] == name:
                recognition_counts[face_id] += 1
                if recognition_counts[face_id] >= FRAMES_TO_CONFIRM and name not in faces_saved_training:
                    person_folder = os.path.join(TRAINING_DIR, name)
                    if os.path.exists(person_folder) and os.path.isdir(person_folder):
                        filename_training = os.path.join(person_folder, f"captured_{int(time.time())}.jpg")
                        cv.imwrite(filename_training, face_image)
                        print(f"TRAINING (CONFIRMED): Riconosciuta come '{name}' per {FRAMES_TO_CONFIRM} frame, salvata in {filename_training}")
                        faces_saved_training.add(name)
            else:
                last_recognition[face_id] = name
                recognition_counts[face_id] = 1
        elif face_id in last_recognition:
            del last_recognition[face_id]
            if face_id in recognition_counts:
                del recognition_counts[face_id]

        label = f"{name} ({confidence:.2f})" if name != "???" else name
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        rect_width = right - left
        font_scale = 0.8
        font_thickness = 1
        text_size = cv.getTextSize(label, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
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