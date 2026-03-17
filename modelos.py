
import numpy as np
from ultralytics import YOLO

def cargar_modelos(path_jugadores, path_bola):
    """
    Cargar los archivos de pesos (.pt) e inicializar los modelos para evitar errores de concurrencia.
    """
    # Cargar los modelos entrenados
    m_players = YOLO(path_jugadores)
    m_ball = YOLO(path_bola)

    # Realizar una predicción vacía (Warmup) para evitar errores de concurrencia
    m_players.predict(np.zeros((128, 128, 3), dtype=np.uint8), verbose=False)
    m_ball.predict(np.zeros((128, 128, 3), dtype=np.uint8), verbose=False)
    
    return m_players, m_ball

def ejecutar_deteccion_jugadores(modelo, frame):
    """
    Ejecutar el seguimiento de jugadores en el frame completo con identificadores persistentes.
    """
    # Usar imgsz=1280 para mantener la precisión en figuras pequeñas o lejanas
    # persist=True permite que ByteTrack asigne un ID único a cada jugador
    return modelo.track(frame, imgsz=1280, persist=True, verbose=False, tracker="bytetrack.yaml")[0]

def ejecutar_deteccion_bola_slice(modelo, recorte, offset_x):
    """
    Detectar la pelota en un recorte (slice) específico y reubicar las coordenadas al plano global.
    """
    # Usar imgsz=960 para que el recorte de 901px mantenga su resolución nativa
    results = modelo.predict(recorte, imgsz=960, verbose=False)[0]
    detecciones = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        
        # Mapear las coordenadas locales del recorte a la resolución original de 2.7K
        # Sumamos el desplazamiento horizontal (offset_x) para situarla correctamente
        detecciones.append([x1 + offset_x, y1, x2 + offset_x, y2, conf])
        
    return detecciones