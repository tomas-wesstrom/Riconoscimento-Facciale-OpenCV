# Importa i moduli necessari
import cv2 as cv  # Libreria OpenCV per l'elaborazione delle immagini
import face_recognition  # Libreria per il riconoscimento facciale
import time  # Modulo per misurare il tempo
import os  # Modulo per interagire con il sistema operativo (es. gestione dei file)
import numpy as np  # Libreria per operazioni numeriche
from PIL import Image  # Python Imaging Library
import shutil  # Modulo per operazioni sui file ad alto livello

# Directory per salvare le immagini delle facce rilevate dalla webcam
FACES_DIR = "detected_faces"
# Directory contenente le immagini utilizzate per l'addestramento del riconoscimento facciale
TRAINING_DIR = "training_images"

# Soglia di tolleranza per confrontare le codifiche delle facce.
FACE_RECOGNITION_TOLERANCE = 0.5

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

# Funzione per caricare i dati di training dinamicamente dalle cartelle.
def load_training_data(training_dir):
    known_face_encodings = []
    known_face_names = {} # **Dizionario per mappare l'indice alla persona (la tua richiesta)**
    name_index = 0
    files_without_faces = []

    # Itera attraverso gli elementi nella directory di training.
    for person_name in os.listdir(training_dir):
        person_path = os.path.join(training_dir, person_name)
        # Verifica se l'elemento corrente è una directory (supponendo che ogni sottocartella sia una persona).
        if os.path.isdir(person_path):
            # Itera attraverso i file all'interno della cartella della persona.
            for filename in os.listdir(person_path):
                # Considera solo i file immagine.
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    filepath = os.path.join(person_path, filename)
                    try:
                        # Carica l'immagine usando la libreria face_recognition.
                        image = face_recognition.load_image_file(filepath)
                        # Ottiene le codifiche delle facce presenti nell'immagine.
                        face_encodings = face_recognition.face_encodings(image)
                        # Se viene trovata almeno una faccia.
                        if len(face_encodings) > 0:
                            # Aggiunge la prima codifica trovata alla lista delle codifiche conosciute.
                            known_face_encodings.append(face_encodings[0])
                            # Associa l'indice della codifica al nome della persona nel dizionario.
                            known_face_names[len(known_face_encodings) - 1] = person_name
                        elif len(face_encodings) == 0:
                            print(f"Nessuna faccia trovata in: {filepath}")
                            files_without_faces.append(filepath)
                    except Exception as e:
                        print(f"Errore nel caricamento o codifica di {filepath}: {e}")
    return known_face_encodings, known_face_names, files_without_faces

# Carica i dati di training.
known_face_encodings, known_face_names, files_to_remove = load_training_data(TRAINING_DIR)
print(f"Caricate {len(known_face_encodings)} codifiche di volti noti da {len(set(known_face_names.values()))} persone.")

# Rimuove le immagini senza volto dalla directory di training.
for filepath in files_to_remove:
    try:
        os.remove(filepath)
        print(f"Rimossa immagine senza volto: {filepath}")
    except OSError as e:
        print(f"Errore nella rimozione di {filepath}: {e}")

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

# Variabili per ottimizzare il processamento dei frame.
last_face_locations = []
last_face_names_with_confidence = []
process_every_n_frames_with_faces = 3
frame_count_with_faces = 0
faces_present = False

# Apri il flusso video dalla webcam.
webcam_video_stream = cv.VideoCapture(0)
desired_fps = 30
webcam_video_stream.set(cv.CAP_PROP_FPS, desired_fps)
actual_fps = webcam_video_stream.get(cv.CAP_PROP_FPS)
print(f"Tentativo FPS: {desired_fps}, FPS effettivi: {actual_fps}")

if not webcam_video_stream.isOpened():
    print("Errore: Impossibile accedere alla camera.")
    exit()
else:
    print("Camera accessibile.")

target_fps = 30
frame_time = 1 / target_fps
frame_counter = 0
skip_frames_no_faces = 0

# Loop principale per l'elaborazione dei frame video.
while True:
    start_time = time.time()
    # Legge un frame dalla webcam.
    ret, frame = webcam_video_stream.read()
    # Se la lettura del frame fallisce, esce dal loop.
    if not ret:
        print("Fallito il recupero del frame")
        break

    frame_counter += 1
    process_frame = False

    # Decide se processare completamente il frame per il rilevamento e la codifica delle facce.
    if not faces_present:
        process_frame = (frame_counter % (skip_frames_no_faces + 1) == 0)
    else:
        process_frame = (frame_count_with_faces % process_every_n_frames_with_faces == 0)

    # Se il flag 'process_frame' è True, esegue il rilevamento e il riconoscimento facciale.
    if process_frame:
        # Ridimensiona il frame per velocizzare il rilevamento facciale.
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Converte l'immagine da BGR (OpenCV) a RGB (face_recognition).
        rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        # Localizza le facce nel frame ridimensionato.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # Calcola le codifiche delle facce trovate.
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names_with_confidence = []

        # Itera attraverso le codifiche delle facce trovate.
        for face_encoding in face_encodings:
            # Confronta la codifica della faccia corrente con le codifiche delle facce conosciute.
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
            name = "???"
            confidence = 0.0

            # Calcola la distanza tra la faccia sconosciuta e le facce conosciute.
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # Se ci sono distanze calcolate e almeno una corrispondenza.
            if len(face_distances) > 0 and np.any(matches):
                # Trova l'indice della faccia conosciuta con la distanza più piccola (migliore corrispondenza).
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    # Ottiene il nome della persona dal dizionario usando l'indice.
                    name = known_face_names.get(best_match_index, "???")
                    # Calcola il livello di confidence come l'inverso della distanza.
                    confidence = 1.0 - face_distances[best_match_index]

            face_names_with_confidence.append((name, confidence))

        # Aggiorna lo stato delle facce presenti.
        if face_locations:
            faces_present = True
            last_face_locations = face_locations
            last_face_names_with_confidence = face_names_with_confidence
            frame_count_with_faces += 1
        else:
            faces_present = False
            frame_count_with_faces = 0
            detection_counts = {}
            last_detection_state = {}
            recognition_counts = {}
            last_recognition = {}
    else:
        # Se il frame non viene processato completamente, utilizza le rilevazioni precedenti.
        face_locations = last_face_locations
        face_names_with_confidence = last_face_names_with_confidence
        if faces_present:
            frame_count_with_faces += 1

    # Mostra i risultati sul frame e gestisce il salvataggio delle facce.
    for i, ((top, right, bottom, left), (name, confidence)) in enumerate(zip(face_locations, face_names_with_confidence)):
        # Scala le coordinate delle facce alla dimensione originale del frame.
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        face_id = i
        # Estrae la regione di interesse (la faccia) dal frame originale.
        face_image = frame[top:bottom, left:right]
        # Determina lo stato del rilevamento: il nome riconosciuto o 'unknown'.
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

        # Crea l'etichetta con il nome e il livello di confidence.
        label = f"{name} ({confidence:.2f})" if name != "???" else name
        # Disegna un rettangolo attorno al volto.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Disegna un rettangolo per il testo del nome.
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        # Aggiunge il testo con il nome e la confidence.
        cv.putText(frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Mostra il frame con i rilevamenti.
    cv.imshow("Rilevamento Facciale", frame)

    # Permette di uscire dal loop premendo il tasto 'q'.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Mantiene il frame rate desiderato.
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)

# Rilascia la webcam e chiude tutte le finestre di OpenCV.
webcam_video_stream.release()
cv.destroyAllWindows()
