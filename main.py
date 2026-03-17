import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Importar los módulos locales
import configuracion
import modelos
import logica

# --- CONFIGURACIÓN DE RUTAS ---
VIDEO_PATH = "video.mp4"
VIDEO_SALIDA = "analisis_tactico.mp4"
MODELO_JUGADORES = "modeloJugadores/best.pt"
MODELO_BOLA = "modeloBola/best.pt"
CONFIG_ROI = "roi_config.txt"
CONFIG_CUADRANTES = "cuadrantes_config.txt" # Nueva ruta

# --- CONFIGURACIÓN DE COLORES (BGR) ---
COL_PISTA    = (200, 200, 200)
COL_JUGADOR  = (255, 144, 30)
COL_BOLA     = (0, 255, 255)
COL_PORTADOR = (0, 255, 0)
COL_ACCION   = (0, 0, 255)
COL_PORTERO  = (255, 0, 255)
COL_ARBITRO  = (0, 165, 255)
COL_PORTERIA = (255, 255, 0)
COL_ZONAS    = (0, 255, 255) # Amarillo para líneas de cuadrantes

def main():
    # 1. Preparar la zona de juego (ROI) y Líneas Tácticas
    # Ahora recibimos dos elementos de la configuración
    puntos_pista, lineas_cuadrantes = configuracion.gestionar_pista(
        VIDEO_PATH, CONFIG_ROI, CONFIG_CUADRANTES, recalibrar=False
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame_ref = cap.read()
    if not ret: return
    
    h_orig, w_orig = frame_ref.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    mascara_pista = configuracion.obtener_mascara(puntos_pista, (h_orig, w_orig))

    # 2. Motores de IA
    m_players, m_ball = modelos.cargar_modelos(MODELO_JUGADORES, MODELO_BOLA)

    # 3. Estados persistentes
    estado_posesion = (None, 0)
    inercia_estado = (None, [0, 0], 0)
    ultimo_portador_id = None

    display_w = 1280
    scale_view = display_w / w_orig
    display_h = int(h_orig * scale_view)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_SALIDA, fourcc, fps, (display_w, display_h))

    with ThreadPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- EJECUCIÓN DE IA EN PARALELO ---
            hilo_p = executor.submit(modelos.ejecutar_deteccion_jugadores, m_players, frame)
            
            ancho_slice = w_orig // 3
            margin = 100
            hilo_b1 = executor.submit(modelos.ejecutar_deteccion_bola_slice, m_ball, 
                                      frame[:, 0 : ancho_slice + margin], 0)
            hilo_b2 = executor.submit(modelos.ejecutar_deteccion_bola_slice, m_ball, 
                                      frame[:, ancho_slice - margin : 2 * ancho_slice + margin], ancho_slice - margin)
            hilo_b3 = executor.submit(modelos.ejecutar_deteccion_bola_slice, m_ball, 
                                      frame[:, 2 * ancho_slice - margin : w_orig], 2 * ancho_slice - margin)

            res_jugadores = hilo_p.result()
            bolas_detectadas = hilo_b1.result() + hilo_b2.result() + hilo_b3.result()

            # --- PROCESAMIENTO DE LÓGICA TÁCTICA ---
            jugadores_pista = logica.filtrar_jugadores_en_pista(res_jugadores, mascara_pista)

            
            portador_id, pos_bola, estado_posesion = logica.validar_posesion(
                bolas_detectadas, jugadores_pista, estado_posesion
            )

            # Si el sistema confirma un portador nuevo, lo grabamos en la memoria
            if portador_id is not None:
                ultimo_portador_id = portador_id

            

            # Actualizamos la llamada para incluir lineas_cuadrantes
            # centro_datos: (fx, fy), radio: puntos_poligono, zonas_activas: [id_zona]
            centro_datos, radio, zonas_activas, inercia_estado = logica.calcular_centro_y_radio(
            pos_bola, 
            jugadores_pista, 
            lineas_cuadrantes, 
            mascara_pista, 
            inercia_estado, 
            portador_id,         # El que la tiene ahora (si existe)
            ultimo_portador_id   # El último que la tuvo (por si acaso)
)
            # --- DIBUJO POR CAPAS ---
            frame_out = frame.copy()
            
            # CAPA 1: Centro de Acción y Zonas Tácticas (Sobre el suelo)
            capa_suelo = frame_out.copy()
            
            # Opcional: Dibujar las líneas de los cuadrantes en el suelo
            for p1, p2 in lineas_cuadrantes:
                cv2.line(capa_suelo, p1, p2, COL_ZONAS, 2, lineType=cv2.LINE_AA)

            if centro_datos:
                fx, fy = centro_datos
                puntos_organicos = radio
                
                # Dibujamos el área de acción adaptada
                cv2.polylines(capa_suelo, [puntos_organicos], isClosed=True, color=COL_ACCION, thickness=4, lineType=cv2.LINE_AA)
                cv2.drawMarker(capa_suelo, (fx, fy), COL_ACCION, cv2.MARKER_CROSS, 40, 3)
                
                # Efecto Parqué: Aplicamos a la imagen original solo donde hay pista
                frame_out = np.where(mascara_pista[:, :, None] == 255, capa_suelo, frame_out)

            # CAPA 2: Límites ROI
            cv2.polylines(frame_out, [puntos_pista], True, COL_PISTA, 2)

            # CAPA 3: Jugadores e IDs
            if res_jugadores.boxes.id is not None:
                classes = res_jugadores.boxes.cls.cpu().numpy()
                names = res_jugadores.names
                ids_ia = res_jugadores.boxes.id.cpu().numpy()

                for j_id, (jx, jy, jbox) in jugadores_pista.items():
                    idx = np.where(ids_ia == j_id)[0][0]
                    nombre_clase = names[int(classes[idx])].lower()

                    color = COL_JUGADOR
                    if j_id == portador_id:
                        color = COL_PORTADOR
                        cv2.putText(frame_out, "PORTADOR", (jbox[0], jbox[1]-25), 0, 0.6, color, 2)
                    elif 'goalkeeper' in nombre_clase: color = COL_PORTERO
                    elif 'referee' in nombre_clase: color = COL_ARBITRO

                    cv2.rectangle(frame_out, (jbox[0], jbox[1]), (jbox[2], jbox[3]), color, 2)
                    cv2.putText(frame_out, f"{nombre_clase.upper()} ID:{j_id}", (jbox[0], jbox[1]-10), 0, 0.5, color, 1)

            # CAPA 4: La Bola
            for b in bolas_detectadas:
                bx, by = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
                cv2.circle(frame_out, (bx, by), 10, COL_BOLA, -1)

            # --- REDIMENSIÓN Y TELEMETRÍA ---
            frame_final = cv2.resize(frame_out, (display_w, display_h))

            # Panel de información táctica
            cv2.rectangle(frame_final, (10, 10), (400, 120), (0,0,0), -1)
            cv2.rectangle(frame_final, (10, 10), (400, 120), (255,255,255), 1)
            
            zona_txt = f"Zona Activa: {zonas_activas[0] + 1 if zonas_activas else 'N/A'}"
            cv2.putText(frame_final, zona_txt, (20, 40), 0, 0.6, COL_ZONAS, 1)
            cv2.putText(frame_final, f"Entidades: {len(jugadores_pista)}", (20, 70), 0, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_final, f"Posesion: {f'JUGADOR {portador_id}' if portador_id else 'LIBRE'}", (20, 100), 0, 0.6, COL_PORTADOR, 2)

            out.write(frame_final)
            cv2.imshow("Analizador Tactico Hockey", frame_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()