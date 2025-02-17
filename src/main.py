import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.detectors import YOLODetector
import cv2
import numpy as np
import math
from filterpy.kalman import KalmanFilter  # Utilisation directe du module

#------------------------------------------------------------------------------
# Fonction pour créer un filtre de Kalman simple pour le suivi
# L'état est défini par [x, y, vx, vy] et la mesure [x, y]
#------------------------------------------------------------------------------
def create_kalman_filter(initial_point):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # État initial [x, y, vx, vy]
    kf.x = np.array([initial_point[0], initial_point[1], 0, 0], dtype=float)
    
    # Matrice de transition (on suppose dt = 1 pour simplifier)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)
    
    # Matrice de mesure : on ne mesure que x et y
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)
    
    # Covariance initiale élevée (incertitude sur la vitesse notamment)
    kf.P *= 1000.
    
    # Bruit de mesure (à ajuster selon votre cas)
    kf.R = np.array([[10, 0],
                     [0, 10]], dtype=float)
    
    # Bruit de processus (peut être ajusté)
    kf.Q = np.eye(4, dtype=float)
    
    return kf

# Initialisation du détecteur YOLO
od = YOLODetector(model_path="yolov8s.pt", use_trt=False)
cap = cv2.VideoCapture("./data/27260-362770008_small.mp4")

# Dictionnaires pour les objets suivis et leurs filtres de Kalman
tracking_objects = {}   # {id: (x, y)}
kalman_filters = {}     # {id: kalman_filter}
missed_frames = {}      # Pour compter le nombre de frames sans mise à jour
track_id = 0            # Identifiant global pour chaque nouvel objet

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Liste des centres des objets détectés dans la frame actuelle
    center_pts_cur_frame = []
    class_ids, scores, boxes = od.detect(frame)
    for box in boxes:
        x_center, y_center, w, h = map(int, box)
        center_pts_cur_frame.append((x_center, y_center))
        # Affichage de la boîte englobante
        cv2.rectangle(frame, (x_center - w//2, y_center - h//2),
                      (x_center + w//2, y_center + h//2), (0, 255, 0), 2)

    # --- Étape 1 : Prédiction pour chaque objet suivi ---
    # On prédit la position de chaque objet en utilisant son filtre de Kalman.
    for obj_id in list(kalman_filters.keys()):
        kalman_filters[obj_id].predict()
        # On met à jour la position estimée avec la prédiction
        tracking_objects[obj_id] = (int(kalman_filters[obj_id].x[0]),
                                    int(kalman_filters[obj_id].x[1]))

    # --- Étape 2 : Association des mesures aux prédictions ---
    assigned_measurements = []  # Pour suivre quelles mesures ont été associées
    # Pour chaque objet suivi, on cherche la mesure la plus proche (sous un certain seuil)
    for obj_id in list(kalman_filters.keys()):
        predicted_pt = (kalman_filters[obj_id].x[0], kalman_filters[obj_id].x[1])
        min_distance = float('inf')
        closest_meas = None
        for pt in center_pts_cur_frame:
            distance = math.hypot(pt[0] - predicted_pt[0], pt[1] - predicted_pt[1])
            if distance < min_distance and distance < 50:  # Seulement si proche (< 50 pixels)
                min_distance = distance
                closest_meas = pt
        if closest_meas is not None:
            # Mise à jour du filtre avec la mesure associée
            kalman_filters[obj_id].update(np.array([closest_meas[0], closest_meas[1]]))
            tracking_objects[obj_id] = (int(kalman_filters[obj_id].x[0]),
                                        int(kalman_filters[obj_id].x[1]))
            assigned_measurements.append(closest_meas)
            missed_frames[obj_id] = 0  # Réinitialiser le compteur de frames manquées
        else:
            # Si aucune mesure n'est associée, on incrémente le compteur
            missed_frames[obj_id] += 1
            if missed_frames[obj_id] > 5:  # Si l'objet n'est pas détecté depuis 5 frames, on le supprime
                del kalman_filters[obj_id]
                del tracking_objects[obj_id]
                del missed_frames[obj_id]

    # --- Étape 3 : Création de nouveaux tracks pour les mesures non associées ---
    for pt in center_pts_cur_frame:
        if pt not in assigned_measurements:
            # Créer un nouveau filtre pour ce nouvel objet
            kf = create_kalman_filter(pt)
            kalman_filters[track_id] = kf
            tracking_objects[track_id] = (int(kf.x[0]), int(kf.x[1]))
            missed_frames[track_id] = 0
            track_id += 1

    # --- Affichage des objets suivis ---
    for obj_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(obj_id), (pt[0], pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Touche Esc pour quitter
        break

cap.release()
cv2.destroyAllWindows()