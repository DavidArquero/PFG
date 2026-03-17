import numpy as np
import cv2

# --- FILTRO DE KALMAN ---
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01 

def aplicar_kalman(x, y):
    medicion = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(medicion)
    prediccion = kalman.predict()
    return int(prediccion[0]), int(prediccion[1])

def filtrar_jugadores_en_pista(resultados_yolo, mascara_pista):
    jugadores_validos = {}
    if resultados_yolo.boxes.id is not None:
        boxes = resultados_yolo.boxes.xyxy.cpu().numpy()
        ids = resultados_yolo.boxes.id.cpu().numpy()
        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            px, py = int((x1 + x2) / 2), int(y2)
            # Verificación de límites de imagen para evitar errores de índice
            h_m, w_m = mascara_pista.shape[:2]
            if mascara_pista[min(py, h_m-1), min(px, w_m-1)] == 255:
                jugadores_validos[int(obj_id)] = (px, int((y1+y2)/2), (x1,y1,x2,y2))
    return jugadores_validos

# --- FUNCIÓN REINTEGRADA: VALIDAR POSESIÓN ---
def validar_posesion(bolas_detectadas, jugadores_dict, estado_posesion, umbral_frames=5):
    """
    Determina qué jugador tiene la bola basándose en la proximidad.
    """
    id_potencial = None
    dist_min = 180 # Píxeles de distancia máxima para considerar posesión
    pos_bola_actual = None
    ultimo_id, contador = estado_posesion

    if bolas_detectadas:
        # Nos quedamos con la detección de bola con mayor confianza
        mejor_bola = max(bolas_detectadas, key=lambda x: x[4])
        temp_pos = (int((mejor_bola[0]+mejor_bola[2])/2), int((mejor_bola[1]+mejor_bola[3])/2))
        pos_bola_actual = temp_pos

        for j_id, (jx, jy, _) in jugadores_dict.items():
            dist = np.sqrt((jx - temp_pos[0])**2 + (jy - temp_pos[1])**2)
            if dist < dist_min:
                dist_min = dist
                id_potencial = j_id

    # Lógica de persistencia para evitar parpadeos de ID
    if id_potencial is not None and id_potencial == ultimo_id:
        contador += 1
    else:
        ultimo_id = id_potencial
        contador = 1

    portador_confirmado = ultimo_id if contador >= umbral_frames else None
    return portador_confirmado, pos_bola_actual, (ultimo_id, contador)

# --- GEOMETRÍA DE CUADRANTES ---

def posicion_respecto_a_linea(punto, linea):
    """ Determina si un punto está a un lado u otro de la línea manual """
    (x, y) = punto
    (p1, p2) = linea
    return (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])

def obtener_zonas_activas(pos_bola, lineas_cuadrantes):
    if not pos_bola or not lineas_cuadrantes:
        return [0]

    idx_zona_bola = 0
    for linea in lineas_cuadrantes:
        if posicion_respecto_a_linea(pos_bola, linea) > 0:
            idx_zona_bola += 1
    return [idx_zona_bola]

# --- PROCESAMIENTO PRINCIPAL ---

def calcular_centro_y_radio(pos_bola, jugadores_dict, lineas_cuadrantes, mascara_pista, inercia_estado, portador_id, ultimo_portador_id):
    ultima_pos, velocidad, frames_perdida = inercia_estado
    UMBRAL_SALTO_MAX = 400

    # --- JERARQUÍA DE SUPERVIVENCIA PARA ENCONTRAR UN PUNTO DE ANCLA ---
    
    # Nivel 1: Tenemos bola real o está en tiempo de inercia (predicción física)
    if pos_bola is None and ultima_pos is not None and frames_perdida < 90:
        pos_bola = (int(ultima_pos[0] + velocidad[0] * 0.95), int(ultima_pos[1] + velocidad[1] * 0.95))
        frames_perdida += 1
    
    # Nivel 2: Si la bola se perdió del todo, usamos al portador identificado actualmente
    elif pos_bola is None and portador_id in jugadores_dict:
        pos_bola = (jugadores_dict[portador_id][0], jugadores_dict[portador_id][1])
        frames_perdida = 0 # Reiniciamos porque tenemos un ancla visual humana
    
    # Nivel 3: Si perdemos al portador actual, usamos el ÚLTIMO portador que recordamos
    elif pos_bola is None and ultimo_portador_id in jugadores_dict:
        pos_bola = (jugadores_dict[ultimo_portador_id][0], jugadores_dict[ultimo_portador_id][1])
        frames_perdida = 0
        
    # Si tras los 3 niveles no hay nada, el sistema se apaga
    if pos_bola is None:
        return None, None, [], (ultima_pos, velocidad, frames_perdida)

    # --- EL RESTO DE LA LÓGICA DE CUADRANTES SE MANTIENE IGUAL ---
    zona_bola = obtener_zonas_activas(pos_bola, lineas_cuadrantes)[0]
    zonas_influencia = [zona_bola - 1, zona_bola, zona_bola + 1]
    
    puntos_interes = [(pos_bola[0], pos_bola[1], 5.0)]
    
    for j_id, (jx, jy, _) in jugadores_dict.items():
        zona_jugador = 0
        for linea in lineas_cuadrantes:
            if posicion_respecto_a_linea((jx, jy), linea) > 0:
                zona_jugador += 1
        
        # Si el jugador está en la zona de la bola o en una zona adyacente
        if zona_jugador in zonas_influencia:
            if zona_jugador == zona_bola:
                peso = 1.5 # Máxima influencia: misma zona
            else:
                peso = 0.6 # Influencia secundaria: zona contigua
            
            puntos_interes.append((jx, jy, peso))

    # --- CÁLCULO DEL CENTRO (Igual que antes pero con nuevos puntos) ---
    sum_w = sum(p[2] for p in puntos_interes)
    cx = int(sum(p[0] * p[2] for p in puntos_interes) / sum_w)
    cy = int(sum(p[1] * p[2] for p in puntos_interes) / sum_w)

    fx, fy = aplicar_kalman(cx, cy)

    # ... (El resto de la generación del polígono se mantiene igual) ...
    radio_ref = 350 
    puntos_poligono = []
    h_m, w_m = mascara_pista.shape[:2]

    for i in range(0, 360, 22):
        angle = np.deg2rad(i)
        px = int(fx + radio_ref * np.cos(angle))
        py = int(fy + radio_ref * np.sin(angle))
        
        iteraciones = 0
        while mascara_pista[max(0, min(py, h_m-1)), max(0, min(px, w_m-1))] == 0 and iteraciones < 5:
            px = int(px * 0.8 + fx * 0.2)
            py = int(py * 0.8 + fy * 0.2)
            iteraciones += 1
        puntos_poligono.append([px, py])

    return (fx, fy), np.array(puntos_poligono, dtype=np.int32), [zona_bola], (ultima_pos, velocidad, frames_perdida)