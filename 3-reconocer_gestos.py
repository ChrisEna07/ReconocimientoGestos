import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time
from collections import deque
import threading
import queue

# ==================== CONFIGURACIÓN OPTIMIZADA ====================
print("="*60)
print("SISTEMA DE RECONOCIMIENTO DE GESTOS - VERSIÓN OPTIMIZADA")
print("="*60)

# Verificar archivos necesarios
archivos_necesarios = ["modelo_gestos.pkl", "scaler.pkl", "label_encoder.pkl"]
archivos_faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]

if archivos_faltantes:
    print("❌ Error: Faltan archivos:")
    for f in archivos_faltantes:
        print(f"   - {f}")
    print("\nPrimero ejecuta: python 2-entrenar_modelo.py")
    exit(1)

# Cargar modelo y preprocesadores
try:
    model = joblib.load("modelo_gestos.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"✓ Modelo cargado: {len(label_encoder.classes_)} gestos")
    print(f"✓ Gestos: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error al cargar: {e}")
    exit(1)

# ==================== CONFIGURACIÓN MEDIAPIPE ====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración optimizada para rendimiento
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,  # 0 = más rápido
    min_detection_confidence=0.3,  # Reducido para más detecciones
    min_tracking_confidence=0.3
)

# ==================== CONFIGURACIÓN DE CÁMARA ====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer

# ==================== PARÁMETROS OPTIMIZADOS ====================
UMBRAL_CONFIANZA = 0.65  # Reducido para más detecciones
HISTORIAL_LEN = 5  # Menos frames para respuesta más rápida
MIN_FRAMES_ESTABLES = 2  # Solo 2 frames estables

# Variables de estado
historial_predicciones = deque(maxlen=HISTORIAL_LEN)
historial_confianzas = deque(maxlen=HISTORIAL_LEN)
gesto_actual = None
frames_estables = 0
frame_count = 0
detecciones_totales = 0

# Variables de rendimiento
tiempos_procesamiento = deque(maxlen=30)
fps = 0
ultimo_tiempo = time.time()
skip_frames = 0  # Para saltar frames si es necesario

# Colores para gestos
COLORES = {
    'pulgar_arriba': (0, 255, 0),
    'pulgar_abajo': (0, 0, 255),
    'ok': (0, 255, 255),
    'victoria': (255, 0, 255),
    'paz': (255, 0, 255),
    'corazon': (0, 255, 255),
    'rock': (255, 165, 0),
    'spiderman': (128, 0, 128),
    'puño': (255, 255, 255),
    'mano_abierta': (255, 255, 0)
}
COLOR_DEFAULT = (100, 255, 100)

# ==================== FUNCIONES OPTIMIZADAS ====================

def predecir_gesto_optimizado(landmarks):
    """Predicción optimizada con caché"""
    try:
        # Normalizar
        landmarks_norm = scaler.transform(landmarks.reshape(1, -1))
        
        # Predecir
        probs = model.predict_proba(landmarks_norm)[0]
        clase_idx = np.argmax(probs)
        confianza = probs[clase_idx]
        
        return clase_idx, confianza
    except:
        return None, 0

def suavizar_prediccion_rapida(clase_idx, confianza):
    """Suavizado rápido con pesos"""
    global gesto_actual, frames_estables
    
    historial_predicciones.append(clase_idx)
    historial_confianzas.append(confianza)
    
    if len(historial_predicciones) < HISTORIAL_LEN:
        return None, 0, False
    
    # Obtener moda del historial (predicción más frecuente)
    from collections import Counter
    prediccion_frecuente = Counter(historial_predicciones).most_common(1)[0][0]
    
    # Calcular confianza promedio
    confianza_promedio = np.mean([c for i, c in zip(historial_predicciones, historial_confianzas) 
                                   if i == prediccion_frecuente])
    
    # Verificar estabilidad
    if prediccion_frecuente == gesto_actual:
        frames_estables += 1
    else:
        gesto_actual = prediccion_frecuente
        frames_estables = 1
    
    estable = frames_estables >= MIN_FRAMES_ESTABLES
    
    return prediccion_frecuente, confianza_promedio, estable

def dibujar_mano_optimizada(frame, hand_landmarks, confianza=None):
    """Dibujado optimizado de la mano"""
    h, w = frame.shape[:2]
    
    # Dibujar solo conexiones principales (más rápido)
    for connection in mp_hands.HAND_CONNECTIONS:
        start = hand_landmarks.landmark[connection[0]]
        end = hand_landmarks.landmark[connection[1]]
        
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        
        cv2.line(frame, start_point, end_point, (100, 255, 100), 1)
    
    # Dibujar puntos clave (solo algunos)
    puntos_clave = [0, 4, 8, 12, 16, 20]  # Muñeca y puntas de dedos
    for idx in puntos_clave:
        lm = hand_landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    
    # Dibujar bounding box simple
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
    
    cv2.rectangle(frame, (x_min-5, y_min-5), (x_max+5, y_max+5), (0, 255, 0), 1)

def dibujar_interfaz_optimizada(frame, gesto, confianza, fps, umbral):
    """Interfaz minimalista pero informativa"""
    h, w = frame.shape[:2]
    
    # Panel semi-transparente más pequeño
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Mostrar gesto si hay detección
    if gesto:
        color = COLORES.get(gesto.lower(), COLOR_DEFAULT)
        
        # Gesto y confianza
        cv2.putText(frame, f"{gesto}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"{confianza*100:.0f}%", (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Barra de confianza
        bar_width = int(250 * confianza)
        cv2.rectangle(frame, (15, 75), (265, 82), (100, 100, 100), -1)
        cv2.rectangle(frame, (15, 75), (15 + bar_width, 82), color, -1)
    else:
        cv2.putText(frame, "Esperando gesto...", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # FPS y umbral
    cv2.putText(frame, f"{int(fps)} fps", (w - 60, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, f"U:{int(umbral*100)}%", (w - 60, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Controles
    cv2.putText(frame, "[+] [-] umbral | [ESC] salir", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return frame

# ==================== LOOP PRINCIPAL ====================
print("\n" + "="*60)
print("🎯 SISTEMA OPTIMIZADO - INICIANDO...")
print("="*60)
print("Controles:")
print("  [+] Aumentar umbral")
print("  [-] Disminuir umbral")
print("  [ESC] Salir")
print("="*60)

# Variables de rendimiento
frame_skip = 0
MAX_SKIP = 1  # Saltar frames si es necesario
ultimo_procesamiento = 0
MIN_PROCESAMIENTO_MS = 30  # Mínimo 30ms entre procesamientos

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer cámara")
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Calcular FPS cada 10 frames
    if frame_count % 10 == 0:
        tiempo_actual = time.time()
        fps = 10 / (tiempo_actual - ultimo_tiempo)
        ultimo_tiempo = tiempo_actual
    
    # Control de velocidad de procesamiento
    tiempo_actual_ms = time.time() * 1000
    procesar_frame = (tiempo_actual_ms - ultimo_procesamiento) >= MIN_PROCESAMIENTO_MS
    
    if procesar_frame:
        ultimo_procesamiento = tiempo_actual_ms
        
        # Procesar con MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        gesto_detectado = None
        confianza_detectada = 0
        
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                # Extraer landmarks
                puntos = []
                for lm in hand.landmark:
                    puntos.extend([lm.x, lm.y, lm.z])
                
                # Predecir
                clase_idx, confianza = predecir_gesto_optimizado(np.array(puntos))
                
                if clase_idx is not None and confianza > UMBRAL_CONFIANZA:
                    # Suavizar predicción
                    clase_suavizada, conf_suavizada, estable = suavizar_prediccion_rapida(clase_idx, confianza)
                    
                    if estable and clase_suavizada is not None:
                        gesto_detectado = label_encoder.inverse_transform([clase_suavizada])[0]
                        confianza_detectada = conf_suavizada
                        detecciones_totales += 1
                
                # Dibujar mano (siempre)
                dibujar_mano_optimizada(frame, hand, confianza)
        else:
            # Reiniciar historial si no hay mano
            historial_predicciones.clear()
            historial_confianzas.clear()
            gesto_actual = None
            frames_estables = 0
        
        # Dibujar interfaz
        frame = dibujar_interfaz_optimizada(frame, gesto_detectado, confianza_detectada, fps, UMBRAL_CONFIANZA)
    
    # Mostrar frame
    cv2.imshow("Reconocimiento de Gestos - Optimizado", frame)
    
    # Manejo de teclas (respuesta rápida)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n🛑 Saliendo...")
        break
    elif key == ord('+') or key == ord('='):  # Aumentar umbral
        UMBRAL_CONFIANZA = min(0.95, UMBRAL_CONFIANZA + 0.05)
        print(f"✓ Umbral: {UMBRAL_CONFIANZA*100:.0f}%")
    elif key == ord('-') or key == ord('_'):  # Disminuir umbral
        UMBRAL_CONFIANZA = max(0.4, UMBRAL_CONFIANZA - 0.05)
        print(f"✓ Umbral: {UMBRAL_CONFIANZA*100:.0f}%")

# ==================== ESTADÍSTICAS FINALES ====================
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("ESTADÍSTICAS FINALES")
print("="*60)
print(f"Frames procesados: {frame_count}")
print(f"Detecciones exitosas: {detecciones_totales}")
if frame_count > 0:
    print(f"Tasa de detección: {(detecciones_totales/frame_count)*100:.1f}%")
print("="*60)
print("✅ Sistema cerrado correctamente")