import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("ENTRENAMIENTO DE MODELO DE RECONOCIMIENTO DE GESTOS")
print("="*60)

# 1. CARGAR DATOS
print("\n[1/8] Cargando dataset...")
try:
    # Intentar cargar con diferentes encodings
    try:
        data = pd.read_csv("gestos.csv", encoding='latin-1')
    except:
        try:
            data = pd.read_csv("gestos.csv", encoding='utf-8')
        except:
            data = pd.read_csv("gestos.csv", encoding='ISO-8859-1')
    
    print(f"✓ Dataset cargado exitosamente")
    print(f"  - Total de muestras: {data.shape[0]}")
    print(f"  - Características: {data.shape[1] - 1}")  # -1 por la columna de gesto
    
except Exception as e:
    print(f"❌ Error al cargar dataset: {e}")
    print("   Asegúrate de que el archivo 'gestos.csv' existe")
    exit(1)

# 2. PREPROCESAMIENTO
print("\n[2/8] Preprocesando datos...")

# Detectar si tiene cabecera o no
if data.iloc[0, 0] == 'x0' or str(data.iloc[0, 0]).startswith('x'):
    print("  - Detectada cabecera en el archivo")
    data = data.iloc[1:].reset_index(drop=True)

# Separar características y etiquetas
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convertir a numérico (por si hay strings)
X = X.astype(float)

# Codificar etiquetas de gestos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_names = label_encoder.classes_

print(f"✓ Gestos detectados: {len(y_names)}")
for i, gesto in enumerate(y_names):
    count = sum(y_encoded == i)
    print(f"  - {gesto}: {count} muestras")

# 3. NORMALIZACIÓN
print("\n[3/8] Normalizando características...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Normalización completada")

# 4. DIVISIÓN DE DATOS
print("\n[4/8] Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded  # Mantiene proporción de clases
)

print(f"✓ Datos divididos:")
print(f"  - Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")

# 5. PROBAR MÚLTIPLES MODELOS
print("\n[5/8] Probando diferentes modelos...")

modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'Red Neuronal': MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

resultados = {}
mejor_modelo = None
mejor_accuracy = 0

for nombre, modelo in modelos.items():
    print(f"\n  Probando {nombre}...")
    try:
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"    ✓ Accuracy: {accuracy*100:.2f}%")
        print(f"    ✓ F1-Score: {f1*100:.2f}%")
        
        # Guardar mejor modelo
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo = modelo
            mejor_nombre = nombre
            
    except Exception as e:
        print(f"    ❌ Error: {e}")

# 6. SELECCIONAR MEJOR MODELO
print("\n[6/8] Seleccionando mejor modelo...")
print(f"✓ Mejor modelo: {mejor_nombre}")
print(f"  - Accuracy: {mejor_accuracy*100:.2f}%")
print(f"  - F1-Score: {resultados[mejor_nombre]['f1_score']*100:.2f}%")

# Entrenar el mejor modelo con todos los datos de entrenamiento
modelo_final = mejor_modelo
modelo_final.fit(X_train, y_train)

# 7. EVALUACIÓN DETALLADA
print("\n[7/8] Evaluando modelo...")
y_pred = modelo_final.predict(X_test)
y_pred_proba = modelo_final.predict_proba(X_test) if hasattr(modelo_final, 'predict_proba') else None

# Reporte de clasificación
print("\n=== REPORTE DE CLASIFICACIÓN POR GESTO ===")
print(classification_report(y_test, y_pred, target_names=y_names))

# Validación cruzada
print("\n=== VALIDACIÓN CRUZADA (5 folds) ===")
cv_scores = cross_val_score(modelo_final, X_scaled, y_encoded, cv=5, scoring='accuracy')
print(f"  Scores: {cv_scores}")
print(f"  Promedio: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Confianza por gesto
if y_pred_proba is not None:
    print("\n=== CONFIANZA PROMEDIO POR GESTO ===")
    for i, gesto in enumerate(y_names):
        mask = y_test == i
        if sum(mask) > 0:
            confianzas = y_pred_proba[mask][:, i]
            confianza_media = confianzas.mean()
            print(f"  {gesto}: {confianza_media*100:.1f}% de confianza promedio")

# 8. GUARDAR MODELO Y PREPROCESADORES
print("\n[8/8] Guardando modelo...")

# Guardar modelo
joblib.dump(modelo_final, "modelo_gestos.pkl")

# Guardar scaler para preprocesamiento en tiempo real
joblib.dump(scaler, "scaler.pkl")

# Guardar label encoder para decodificar predicciones
joblib.dump(label_encoder, "label_encoder.pkl")

print("✓ Modelo guardado como 'modelo_gestos.pkl'")
print("✓ Scaler guardado como 'scaler.pkl'")
print("✓ Label encoder guardado como 'label_encoder.pkl'")

# 9. GENERAR MATRIZ DE CONFUSIÓN
print("\n=== GENERANDO MATRIZ DE CONFUSIÓN ===")
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=y_names,
            yticklabels=y_names)
plt.title('Matriz de Confusión - Reconocimiento de Gestos')
plt.ylabel('Gesto Real')
plt.xlabel('Gesto Predicho')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=100)
print("✓ Matriz de confusión guardada como 'matriz_confusion.png'")

# 10. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
if isinstance(modelo_final, (RandomForestClassifier, GradientBoostingClassifier)):
    print("\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
    importancias = modelo_final.feature_importances_
    
    # Mostrar top 10 características más importantes
    top_n = min(10, len(importancias))
    indices_top = np.argsort(importancias)[-top_n:][::-1]
    
    print(f"Top {top_n} características más importantes:")
    for i, idx in enumerate(indices_top):
        # Determinar qué punto y coordenada es
        punto = idx // 3
        coordenada = ['X', 'Y', 'Z'][idx % 3]
        print(f"  {i+1}. Punto {punto} - Coordenada {coordenada}: {importancias[idx]:.4f}")
    
    # Visualizar importancia
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importancias)), importancias)
    plt.title('Importancia de Características')
    plt.xlabel('Características (21 puntos x 3 coordenadas)')
    plt.ylabel('Importancia')
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png', dpi=100)
    print("✓ Gráfico de importancia guardado como 'importancia_caracteristicas.png'")

# 11. RECOMENDACIONES
print("\n" + "="*60)
print("RECOMENDACIONES Y ANÁLISIS")
print("="*60)

# Verificar cantidad de muestras por gesto
muestras_por_gesto = [sum(y_encoded == i) for i in range(len(y_names))]
min_muestras = min(muestras_por_gesto)

if min_muestras < 50:
    print("\n⚠️ ADVERTENCIA: Algunos gestos tienen menos de 50 muestras:")
    for i, gesto in enumerate(y_names):
        if muestras_por_gesto[i] < 50:
            print(f"  - {gesto}: {muestras_por_gesto[i]} muestras")
    print("  Recomendación: Capturar más muestras de estos gestos para mejorar precisión")

# Verificar precisión general
if mejor_accuracy < 0.8:
    print("\n⚠️ ADVERTENCIA: La precisión general es baja (<80%)")
    print("  Recomendaciones:")
    print("  1. Capturar más muestras por gesto")
    print("  2. Asegurar consistencia en los gestos capturados")
    print("  3. Capturar muestras en diferentes condiciones (luz, distancia, ángulo)")
    print("  4. Revisar la matriz de confusión para identificar gestos confusos")

# Verificar confianza por gesto
if y_pred_proba is not None:
    confianzas_bajas = []
    for i, gesto in enumerate(y_names):
        mask = y_test == i
        if sum(mask) > 0:
            confianzas = y_pred_proba[mask][:, i]
            confianza_media = confianzas.mean()
            if confianza_media < 0.7:
                confianzas_bajas.append((gesto, confianza_media))
    
    if confianzas_bajas:
        print("\n⚠️ GESTOS CON CONFIANZA BAJA (<70%):")
        for gesto, conf in confianzas_bajas:
            print(f"  - {gesto}: {conf*100:.1f}%")
        print("  Recomendación: Capturar más muestras de estos gestos con mayor variabilidad")

# Mostrar resumen final
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"✓ Dataset: {data.shape[0]} muestras, {len(y_names)} gestos")
print(f"✓ Mejor modelo: {mejor_nombre}")
print(f"✓ Precisión general: {mejor_accuracy*100:.2f}%")
print(f"✓ Archivos generados:")
print("  - modelo_gestos.pkl")
print("  - scaler.pkl")
print("  - label_encoder.pkl")
print("  - matriz_confusion.png")
if isinstance(modelo_final, (RandomForestClassifier, GradientBoostingClassifier)):
    print("  - importancia_caracteristicas.png")
print("="*60)
print("\n🎉 ¡Entrenamiento completado exitosamente!")
print("   Ahora puedes ejecutar el reconocimiento en tiempo real")