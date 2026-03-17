import cv2
import numpy as np
import os

def gestionar_pista(video_path, config_roi, config_cuadrantes, config_saque, recalibrar=False):
    """
    Orquestador de la configuración: Carga los archivos de texto o lanza el calibrador
    si los archivos no existen o se solicita recalibración manual.
    """
    puntos_roi = []
    lineas_cuadrantes = []
    punto_saque = None

    # --- 1. CARGA DE PERSISTENCIA ---
    # Solo intentamos cargar si NO se ha solicitado una recalibración manual desde cero
    if not recalibrar:
        if os.path.exists(config_roi):
            with open(config_roi, "r") as f:
                puntos_roi = [list(map(int, l.strip().split(","))) for l in f]
        
        if os.path.exists(config_cuadrantes):
            with open(config_cuadrantes, "r") as f:
                for l in f:
                    coords = list(map(int, l.strip().split(",")))
                    lineas_cuadrantes.append(((coords[0], coords[1]), (coords[2], coords[3])))
        
        if os.path.exists(config_saque):
            with open(config_saque, "r") as f:
                linea = f.readline()
                if linea:
                    punto_saque = list(map(int, linea.strip().split(",")))

    # --- 2. VALIDACIÓN Y LANZAMIENTO DEL EDITOR ---
    # Se entra al calibrador si: se pide manual, o si falta cualquiera de los archivos
    if recalibrar or not puntos_roi or not lineas_cuadrantes or not punto_saque:
        # Si es recalibración manual, forzamos que las listas empiecen vacías (Borrón y cuenta nueva)
        if recalibrar:
            puntos_roi, lineas_cuadrantes, punto_saque = [], [], None

        puntos_roi, lineas_cuadrantes, punto_saque = _calibrar_todo(
            video_path, config_roi, config_cuadrantes, config_saque, puntos_roi, lineas_cuadrantes, punto_saque
        )

    return np.array(puntos_roi, dtype=np.int32), lineas_cuadrantes, punto_saque

def _calibrar_todo(video_path, config_roi, config_cuadrantes, config_saque, roi_previo, lineas_previas, saque_previo):
    """
    Interfaz gráfica de calibración con máquina de estados: ROI -> CUADRANTES -> SAQUE.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return [], [], None

    # Re-escalado para visualización cómoda en monitores estándar
    h_orig, w_orig = frame.shape[:2]
    scale = 1280 / w_orig
    frame_res = cv2.resize(frame, (1280, int(h_orig * scale)))

    puntos_roi = list(roi_previo)
    lineas_cuadrantes = list(lineas_previas)
    punto_saque = saque_previo
    punto_temporal = None 
    
    # Determinamos el modo inicial según los datos cargados
    modo = "ROI"
    if len(puntos_roi) > 3: modo = "CUADRANTES"
    if len(lineas_cuadrantes) >= 1: modo = "SAQUE"

    def mouse_cb(event, x, y, flags, param):
        """ Manejador de eventos del ratón """
        nonlocal modo, punto_temporal, punto_saque
        if event == cv2.EVENT_LBUTTONDOWN:
            # Traducir clic de 1280px a resolución original del vídeo
            val = (int(x / scale), int(y / scale))
            
            if modo == "ROI":
                puntos_roi.append(val)
            elif modo == "CUADRANTES":
                if punto_temporal is None:
                    punto_temporal = val
                else:
                    lineas_cuadrantes.append((punto_temporal, val))
                    punto_temporal = None
            elif modo == "SAQUE":
                punto_saque = val

    cv2.namedWindow("CALIBRADOR TACTICO")
    cv2.setMouseCallback("CALIBRADOR TACTICO", mouse_cb)

    while True:
        img_draw = frame_res.copy()
        
        # --- DIBUJO DE ELEMENTOS ---
        # 1. Dibujar ROI
        pts_view = [(int(p[0] * scale), int(p[1] * scale)) for p in puntos_roi]
        if len(pts_view) > 1:
            color_roi = (0, 255, 0) if modo == "ROI" else (0, 150, 0)
            cv2.polylines(img_draw, [np.array(pts_view)], (modo != "ROI"), color_roi, 2)
        for p_v in pts_view:
            cv2.circle(img_draw, p_v, 4, (0, 255, 0), -1)

        # 2. Dibujar Cuadrantes
        for p1, p2 in lineas_cuadrantes:
            cv2.line(img_draw, (int(p1[0]*scale), int(p1[1]*scale)), 
                               (int(p2[0]*scale), int(p2[1]*scale)), (0, 255, 255), 2)
        if punto_temporal:
            cv2.circle(img_draw, (int(punto_temporal[0]*scale), int(punto_temporal[1]*scale)), 5, (0, 255, 255), -1)

        # 3. Dibujar Punto de Saque
        if punto_saque:
            p_s_v = (int(punto_saque[0]*scale), int(punto_saque[1]*scale))
            cv2.drawMarker(img_draw, p_s_v, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # --- TEXTOS DE INTERFAZ ---
        cv2.putText(img_draw, f"MODO: {modo}", (20, 40), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(img_draw, "ENTER: Validar/Sig | Z: Deshacer | R: Limpiar Modo | ESC: Salir", (20, 70), 0, 0.5, (200, 255, 255), 1)

        cv2.imshow("CALIBRADOR TACTICO", img_draw)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: # ESC
            break
        elif key == ord("r"): # Reiniciar modo actual
            if modo == "ROI": puntos_roi = []
            elif modo == "CUADRANTES": lineas_cuadrantes = []; punto_temporal = None
            elif modo == "SAQUE": punto_saque = None
        elif key == ord("z") or key == 8: # Deshacer
            if modo == "SAQUE": 
                if punto_saque: punto_saque = None
                else: modo = "CUADRANTES"
            elif modo == "CUADRANTES":
                if punto_temporal: punto_temporal = None
                elif lineas_cuadrantes: lineas_cuadrantes.pop()
                else: modo = "ROI"
            elif puntos_roi: puntos_roi.pop()
        elif key == 13: # ENTER: Avanzar o Guardar
            if modo == "ROI" and len(puntos_roi) > 3:
                modo = "CUADRANTES"
            elif modo == "CUADRANTES" and len(lineas_cuadrantes) >= 1:
                modo = "SAQUE"
            elif modo == "SAQUE" and punto_saque is not None:
                # PERSISTENCIA: Guardamos todo en archivos de texto
                with open(config_roi, "w") as f:
                    for p in puntos_roi: f.write(f"{p[0]},{p[1]}\n")
                with open(config_cuadrantes, "w") as f:
                    for p1, p2 in lineas_cuadrantes:
                        f.write(f"{p1[0]},{p1[1]},{p2[0]},{p2[1]}\n")
                with open(config_saque, "w") as f:
                    f.write(f"{punto_saque[0]},{punto_saque[1]}\n")
                break

    cv2.destroyAllWindows()
    return puntos_roi, lineas_cuadrantes, punto_saque

def obtener_mascara(puntos, shape_original):
    """
    Genera una imagen binaria (máscara) donde el interior del ROI es blanco (255)
    y el exterior es negro (0). Fundamental para filtrar detecciones fuera de la pista.
    """
    mask = np.zeros(shape_original[:2], dtype=np.uint8)
    if puntos is not None and len(puntos) > 3:
        # Convertimos a formato int32 que es lo que requiere fillPoly
        pts = np.array(puntos, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask