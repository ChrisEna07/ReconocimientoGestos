import cv2
import mediapipe as mp
import csv
import os
import time
from collections import deque

# ==================== CONFIGURACIÓN OPTIMIZADA ====================
# Configuración de MediaPipe para máximo rendimiento
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración optimizada para velocidad
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,  # 0 = menor complejidad, más rápido (0, 1, 2)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar cámara con configuración optimizada
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia

# ==================== VARIABLES GLOBALES ====================
# Buffer de landmarks para captura rápida
landmark_buffer = deque(maxlen=5)
ultima_captura = 0
delay_entre_capturas = 0.1  # 100ms entre capturas (10 fps)

# Variables de estabilidad simplificadas
contador_estabilidad = 0
UMBRAL_ESTABILIDAD = 2  # Solo 2 frames para estabilidad

# ==================== FUNCIONES OPTIMIZADAS ====================

def detectar_mano_rapida(hand_landmarks):
    """Extrae landmarks rápidamente sin procesamiento extra"""
    puntos = []
    for lm in hand_landmarks.landmark:
        puntos.extend([lm.x, lm.y, lm.z])
    return puntos

def verificar_estabilidad(puntos_actuales):
    """Verificación rápida de estabilidad"""
    global contador_estabilidad
    
    if len(landmark_buffer) > 0:
        puntos_anteriores = landmark_buffer[-1]
        # Calcular diferencia rápida (sin numpy para mayor velocidad)
        diferencia = sum(abs(a - b) for a, b in zip(puntos_actuales, puntos_anteriores))
        
        if diferencia < 0.015:  # Umbral bajo para respuesta rápida
            contador_estabilidad += 1
            return contador_estabilidad >= UMBRAL_ESTABILIDAD
        else:
            contador_estabilidad = 0
            return False
    else:
        landmark_buffer.append(puntos_actuales)
        return False

def guardar_muestra_rapida(writer, puntos, gesto, archivo):
    """Guarda muestra de forma optimizada"""
    writer.writerow(puntos + [gesto])
    archivo.flush()  # Forzar escritura inmediata

def dibujar_interfaz_rapida(frame, gesto, muestras, objetivo, modo_auto, fps):
    """Dibuja interfaz minimalista para máximo rendimiento"""
    # Fondo semi-transparente reducido
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (350, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Información esencial
    cv2.putText(frame, f"{gesto}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, f"{muestras}/{objetivo}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Barra de progreso simple
    progress = int((muestras / objetivo) * 300) if objetivo > 0 else 0
    cv2.rectangle(frame, (20, 85), (320, 95), (100, 100, 100), -1)
    cv2.rectangle(frame, (20, 85), (20 + progress, 95), (0, 255, 0), -1)
    
    # Indicadores rápidos
    if modo_auto:
        cv2.putText(frame, "AUTO", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(frame, "MANUAL", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # FPS
    cv2.putText(frame, f"{int(fps)}fps", (frame.shape[1] - 60, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame

def dibujar_mano_rapida(frame, hand_landmarks):
    """Dibuja solo puntos clave para mayor velocidad"""
    # Dibujar solo conexiones principales
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        
        start_coords = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
        end_coords = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
        
        cv2.line(frame, start_coords, end_coords, (100, 255, 100), 1)
    
    # Dibujar solo algunos puntos clave
    puntos_clave = [0, 4, 8, 12, 16, 20]  # Muñeca y puntas de dedos
    for idx in puntos_clave:
        lm = hand_landmarks.landmark[idx]
        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

# ==================== FUNCIÓN PRINCIPAL DE CAPTURA ====================

def capturar_gesto_rapido():
    """Versión optimizada para máxima velocidad"""
    global ultima_captura, landmark_buffer, contador_estabilidad
    
    # Solicitar datos del gesto
    print("\n" + "="*50)
    gesto = input("\nNombre del gesto: ").strip()
    
    if gesto.lower() == 'salir':
        return False
    
    while not gesto:
        gesto = input("Nombre del gesto: ").strip()
    
    objetivo_input = input(f"Muestras para '{gesto}' (default 200): ").strip()
    objetivo = int(objetivo_input) if objetivo_input.isdigit() else 200
    
    print(f"\n✓ Capturando: {gesto}")
    print("Controles: [s]Guardar [a]Auto [ESC]Salir")
    print("="*50)
    
    # Variables de control
    muestras = 0
    modo_auto = False
    fps = 0
    frame_count = 0
    tiempo_anterior = time.time()
    
    # Buffer para captura rápida
    captura_pendiente = False
    tiempo_captura = 0
    
    # Limpiar buffers
    landmark_buffer.clear()
    contador_estabilidad = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Calcular FPS
        if frame_count % 10 == 0:
            tiempo_actual = time.time()
            fps = 10 / (tiempo_actual - tiempo_anterior) if tiempo_actual != tiempo_anterior else 0
            tiempo_anterior = tiempo_actual
        
        # Procesar frame con MediaPipe (optimizado)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Detectar manos
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                # Extraer landmarks rápidamente
                puntos = detectar_mano_rapida(hand)
                
                # Verificar estabilidad rápida
                estable = verificar_estabilidad(puntos)
                
                if estable:
                    landmark_buffer.append(puntos)
                    
                    # Captura automática
                    if modo_auto and not captura_pendiente:
                        tiempo_actual = time.time()
                        if tiempo_actual - ultima_captura >= delay_entre_capturas:
                            guardar_muestra_rapida(writer, puntos, gesto, archivo)
                            muestras += 1
                            ultima_captura = tiempo_actual
                            captura_pendiente = True
                            tiempo_captura = tiempo_actual
                            print(f"✓ {muestras}/{objetivo}")
                    
                    # Feedback visual rápido
                    if captura_pendiente and time.time() - tiempo_captura < 0.1:
                        cv2.putText(frame, "CAPTURADO!", (frame.shape[1]//2 - 50, frame.shape[0]//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        captura_pendiente = False
                
                # Dibujar mano simplificado
                dibujar_mano_rapida(frame, hand)
                
                # Indicador de estabilidad rápido
                if estable:
                    cv2.putText(frame, "●", (frame.shape[1] - 30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "○", (frame.shape[1] - 30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Reiniciar estabilidad si no hay mano
            landmark_buffer.clear()
            contador_estabilidad = 0
        
        # Dibujar interfaz minimalista
        frame = dibujar_interfaz_rapida(frame, gesto, muestras, objetivo, modo_auto, fps)
        
        # Mostrar frame
        cv2.imshow("Captura Rapida", frame)
        
        # Manejo de teclas (respuesta rápida)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n❌ Captura cancelada")
            return True
        
        elif key == ord('s') and not modo_auto:  # Guardar manual
            if result.multi_hand_landmarks:
                puntos = detectar_mano_rapida(result.multi_hand_landmarks[0])
                guardar_muestra_rapida(writer, puntos, gesto, archivo)
                muestras += 1
                print(f"✓ {muestras}/{objetivo}")
                
                # Feedback visual
                cv2.putText(frame, "GUARDADO!", (frame.shape[1]//2 - 50, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Captura Rapida", frame)
                cv2.waitKey(100)
        
        elif key == ord('a'):  # Alternar modo automático
            modo_auto = not modo_auto
            if modo_auto:
                print("✅ Modo automático activado")
                ultima_captura = time.time()
                landmark_buffer.clear()
                contador_estabilidad = 0
            else:
                print("❌ Modo automático desactivado")
        
        # Verificar si completó
        if muestras >= objetivo:
            print(f"\n🎉 Completado: {muestras} muestras")
            cv2.putText(frame, "COMPLETADO!", (frame.shape[1]//2 - 60, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Captura Rapida", frame)
            cv2.waitKey(1000)
            return True
    
    return True

# ==================== CONFIGURACIÓN INICIAL ====================

# Verificar archivo CSV
archivo_existe = os.path.isfile("gestos.csv")
archivo = open("gestos.csv", "a", newline="")
writer = csv.writer(archivo)

# Crear cabecera si es necesario
if not archivo_existe:
    cabecera = []
    for i in range(21):
        cabecera.extend([f'x{i}', f'y{i}', f'z{i}'])
    cabecera.append('gesto')
    writer.writerow(cabecera)
    print("✓ Archivo CSV creado")

print("\n" + "="*50)
print("CAPTURA RÁPIDA DE GESTOS")
print("="*50)
print("✓ Optimizado para máxima velocidad")
print("✓ Modelo de mano: baja complejidad")
print("✓ Procesamiento optimizado")
print("="*50)

# Capturar gestos
total_gestos = 0
while True:
    resultado = capturar_gesto_rapido()
    if not resultado:
        break
    
    total_gestos += 1
    
    # Preguntar por otro gesto
    otro = input("\n¿Otro gesto? (s/n): ").strip().lower()
    if otro != 's':
        break

# ==================== LIMPIEZA ====================
cap.release()
archivo.close()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("RESUMEN FINAL")
print("="*50)
print(f"✓ Gestos capturados: {total_gestos}")
print(f"✓ Archivo: gestos.csv")
print("="*50)
print("\n✅ ¡Listo! Entrena con:")
print("   python 2-entrenar_modelo.py")