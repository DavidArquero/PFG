import cv2
import numpy as np
import os

def gestionar_pista(video_path, config_roi, config_cuadrantes, recalibrar=False):
    puntos_roi = []
    lineas_cuadrantes = [] # Ahora guardaremos pares de puntos [(p1, p2), (p3, p4)...]

    if os.path.exists(config_roi):
        with open(config_roi, "r") as f:
            puntos_roi = [list(map(int, l.strip().split(","))) for l in f]
    
    if os.path.exists(config_cuadrantes):
        with open(config_cuadrantes, "r") as f:
            # Cargamos pares de puntos: x1,y1,x2,y2
            for l in f:
                coords = list(map(int, l.strip().split(",")))
                lineas_cuadrantes.append(((coords[0], coords[1]), (coords[2], coords[3])))

    if recalibrar or not puntos_roi or not lineas_cuadrantes:
        puntos_roi, lineas_cuadrantes = _calibrar_todo(video_path, config_roi, config_cuadrantes, puntos_roi, lineas_cuadrantes)

    return np.array(puntos_roi, dtype=np.int32), lineas_cuadrantes

def _calibrar_todo(video_path, config_roi, config_cuadrantes, roi_previo, lineas_previas):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return [], []

    h_orig, w_orig = frame.shape[:2]
    scale = 1280 / w_orig
    frame_res = cv2.resize(frame, (1280, int(h_orig * scale)))

    puntos_roi = list(roi_previo)
    lineas_cuadrantes = list(lineas_previas)
    punto_temporal = None # Para guardar el primer clic de una línea
    
    modo = "CUADRANTES" if len(puntos_roi) > 3 else "ROI"

    def mouse_cb(event, x, y, flags, param):
        nonlocal modo, punto_temporal
        if event == cv2.EVENT_LBUTTONDOWN:
            val = (int(x / scale), int(y / scale))
            if modo == "ROI":
                puntos_roi.append(val)
            else:
                if punto_temporal is None:
                    punto_temporal = val
                else:
                    lineas_cuadrantes.append((punto_temporal, val))
                    punto_temporal = None

    cv2.namedWindow("CALIBRADOR TACTICO")
    cv2.setMouseCallback("CALIBRADOR TACTICO", mouse_cb)

    while True:
        img_draw = frame_res.copy()
        
        # Dibujo ROI (Verde)
        pts_view = [(int(p[0] * scale), int(p[1] * scale)) for p in puntos_roi]
        if len(pts_view) > 1:
            cv2.polylines(img_draw, [np.array(pts_view)], (modo == "CUADRANTES"), (0, 255, 0), 2)
        for p_v in pts_view:
            cv2.circle(img_draw, p_v, 5, (0, 255, 0), -1) # Círculo relleno de verde

        # Dibujo Líneas Cuadrantes (Amarillo)
        for p1, p2 in lineas_cuadrantes:
            cv2.line(img_draw, (int(p1[0]*scale), int(p1[1]*scale)), 
                               (int(p2[0]*scale), int(p2[1]*scale)), (0, 255, 255), 2)
        
        # Dibujo punto suelto (el primer clic de una línea)
        if punto_temporal:
            cv2.circle(img_draw, (int(punto_temporal[0]*scale), int(punto_temporal[1]*scale)), 5, (0, 255, 255), -1)

        # Interfaz
        txt = f"MODO: {modo} | Lineas: {len(lineas_cuadrantes)}"
        cv2.putText(img_draw, txt, (20, 40), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(img_draw, "ROI: Clics libres | LINEAS: 2 Clics por linea | ENTER: Sig/Fin", (20, 70), 0, 0.5, (255, 255, 255), 1)

        cv2.imshow("CALIBRADOR TACTICO", img_draw)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("z") or key == 8:
            if punto_temporal: punto_temporal = None
            elif modo == "CUADRANTES":
                if lineas_cuadrantes: lineas_cuadrantes.pop()
                else: modo = "ROI"
            elif puntos_roi: puntos_roi.pop()
        
        elif key == 13: # ENTER
            if modo == "ROI" and len(puntos_roi) > 3:
                modo = "CUADRANTES"
            elif modo == "CUADRANTES" and len(lineas_cuadrantes) >= 1:
                with open(config_roi, "w") as f:
                    for p in puntos_roi: f.write(f"{p[0]},{p[1]}\n")
                with open(config_cuadrantes, "w") as f:
                    for p1, p2 in lineas_cuadrantes:
                        f.write(f"{p1[0]},{p1[1]},{p2[0]},{p2[1]}\n")
                break

    cv2.destroyAllWindows()
    return puntos_roi, lineas_cuadrantes

def obtener_mascara(puntos, shape_original):
    """
    Genera una máscara binaria a partir del polígono definido por los puntos.
    """
    mask = np.zeros(shape_original[:2], dtype=np.uint8)
    if puntos is not None and len(puntos) > 3:
        # Aseguramos que los puntos estén en formato correcto para OpenCV
        pts = np.array(puntos, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask