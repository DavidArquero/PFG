import cv2
import numpy as np
import os

def gestionar_pista(video_path, config_roi, config_cuadrantes, recalibrar=False):
    """
    Orquestador de la configuración: Carga los archivos de texto o lanza el calibrador
    si los archivos no existen o se solicita recalibración manual.
    """
    puntos_roi = []
    lineas_cuadrantes = []

    # --- 1. CARGA DE PERSISTENCIA ---
    if not recalibrar:
        if os.path.exists(config_roi):
            with open(config_roi, "r") as f:
                puntos_roi = [list(map(int, l.strip().split(","))) for l in f]
        
        if os.path.exists(config_cuadrantes):
            with open(config_cuadrantes, "r") as f:
                for l in f:
                    coords = list(map(int, l.strip().split(",")))
                    lineas_cuadrantes.append(((coords[0], coords[1]), (coords[2], coords[3])))

    # --- 2. VALIDACIÓN Y LANZAMIENTO DEL EDITOR ---
    if recalibrar or not puntos_roi or not lineas_cuadrantes:
        if recalibrar:
            puntos_roi, lineas_cuadrantes = [], []

        puntos_roi, lineas_cuadrantes = _calibrar_todo(
            video_path, config_roi, config_cuadrantes, puntos_roi, lineas_cuadrantes
        )

    return np.array(puntos_roi, dtype=np.int32), lineas_cuadrantes

def _calibrar_todo(video_path, config_roi, config_cuadrantes, roi_previo, lineas_previas):
    """
    Interfaz gráfica de calibración: ROI -> CUADRANTES.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return [], []

    h_orig, w_orig = frame.shape[:2]
    scale = 1280 / w_orig
    frame_res = cv2.resize(frame, (1280, int(h_orig * scale)))

    puntos_roi = list(roi_previo)
    lineas_cuadrantes = list(lineas_previas)
    punto_temporal = None 
    
    modo = "ROI"
    if len(puntos_roi) > 3: modo = "CUADRANTES"

    def mouse_cb(event, x, y, flags, param):
        nonlocal modo, punto_temporal
        if event == cv2.EVENT_LBUTTONDOWN:
            val = (int(x / scale), int(y / scale))
            if modo == "ROI":
                puntos_roi.append(val)
            elif modo == "CUADRANTES":
                if punto_temporal is None:
                    punto_temporal = val
                else:
                    lineas_cuadrantes.append((punto_temporal, val))
                    punto_temporal = None

    cv2.namedWindow("CALIBRADOR TACTICO")
    cv2.setMouseCallback("CALIBRADOR TACTICO", mouse_cb)

    while True:
        img_draw = frame_res.copy()
        
        # Dibujar ROI
        pts_view = [(int(p[0] * scale), int(p[1] * scale)) for p in puntos_roi]
        if len(pts_view) > 1:
            color_roi = (0, 255, 0) if modo == "ROI" else (0, 150, 0)
            cv2.polylines(img_draw, [np.array(pts_view)], (modo != "ROI"), color_roi, 2)
        for p_v in pts_view:
            cv2.circle(img_draw, p_v, 4, (0, 255, 0), -1)

        # Dibujar Cuadrantes
        for p1, p2 in lineas_cuadrantes:
            cv2.line(img_draw, (int(p1[0]*scale), int(p1[1]*scale)), 
                               (int(p2[0]*scale), int(p2[1]*scale)), (0, 255, 255), 2)
        if punto_temporal:
            cv2.circle(img_draw, (int(punto_temporal[0]*scale), int(punto_temporal[1]*scale)), 5, (0, 255, 255), -1)

        cv2.putText(img_draw, f"MODO: {modo}", (20, 40), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(img_draw, "ENTER: Sig/Guardar | Z: Deshacer | R: Limpiar | ESC: Salir", (20, 70), 0, 0.5, (200, 255, 255), 1)

        cv2.imshow("CALIBRADOR TACTICO", img_draw)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: break
        elif key == ord("r"):
            if modo == "ROI": puntos_roi = []
            else: lineas_cuadrantes = []; punto_temporal = None
        elif key == ord("z") or key == 8:
            if punto_temporal: punto_temporal = None
            elif modo == "CUADRANTES":
                if lineas_cuadrantes: lineas_cuadrantes.pop()
                else: modo = "ROI"
            elif puntos_roi: puntos_roi.pop()
        elif key == 13:
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
    mask = np.zeros(shape_original[:2], dtype=np.uint8)
    if puntos is not None and len(puntos) > 3:
        pts = np.array(puntos, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask