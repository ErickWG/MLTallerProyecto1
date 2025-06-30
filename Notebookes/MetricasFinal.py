# %%
import pandas as pd
import numpy as np
from river import anomaly
from river import preprocessing
import pickle
import os
from datetime import datetime
import warnings
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("✅ Librerías importadas correctamente")

# %% [markdown]
# # # 1. CONFIGURACIÓN DE RUTAS Y CARGA DE MODELO

# %%
# Rutas de archivos
MODELS_PATH = r"C:\Users\User\Desktop\TESIS\CodigoGithub\MLTallerProyecto1\src\Modelos"
EVALUATION_CSV_PATH = r"C:\Users\User\Desktop\TESIS\NuevoDataSet\DataSetFinalProbarMatriz.csv"  # 🔧 CAMBIAR ESTA RUTA
RESULTADO = r"C:\Users\User\Desktop\TESIS\CodigoGithub\MLTallerProyecto1\src\Resultados_API"

# Verificar que el directorio de modelos existe
if not os.path.exists(MODELS_PATH):
    print(f"❌ Error: Directorio de modelos no encontrado: {MODELS_PATH}")
    exit()

print(f"📁 Directorio de modelos: {MODELS_PATH}")

# %% [markdown]
# # 2. CARGAR MODELO, SCALER Y CONFIGURACIÓN

# %%
print("\n🔄 Cargando modelo entrenado...")

# Cargar modelo
modelo_path = os.path.join(MODELS_PATH, "modelo_general.pkl")
with open(modelo_path, 'rb') as f:
    modelo_cargado = pickle.load(f)

# Cargar scaler
scaler_path = os.path.join(MODELS_PATH, "scaler_general.pkl")
with open(scaler_path, 'rb') as f:
    scaler_cargado = pickle.load(f)

# Cargar configuración
config_path = os.path.join(MODELS_PATH, "config_modelo_general.pkl")
with open(config_path, 'rb') as f:
    config_cargado = pickle.load(f)

print("✅ Modelo cargado exitosamente")

# Mostrar información del modelo
print(f"\n📊 INFORMACIÓN DEL MODELO CARGADO:")
print(f"🎯 Umbral global: {config_cargado['umbral_global']:.4f}")
print(f"🌍 Países en entrenamiento: {config_cargado['paises_entrenamiento']}")
print(f"📈 Registros de entrenamiento: {config_cargado['registros_entrenamiento']}")
print(f"📅 Fecha de entrenamiento: {config_cargado['fecha_entrenamiento']}")
print(f"🌳 Número de árboles: {config_cargado['n_trees']}")
print(f"📏 Altura de árboles: {config_cargado['tree_height']}")

# Extraer configuraciones
umbral_global = config_cargado['umbral_global']
stats_dict  = config_cargado['stats_por_pais']
parametros_features = config_cargado['parametros_features']
contexto_historico = config_cargado['contexto_historico']


# Configurar parámetros de features
PESO_MINUTOS_NORMAL = parametros_features['peso_minutos_normal']
PESO_MINUTOS_EXTREMOS = parametros_features['peso_minutos_extremos']
UMBRAL_MINUTOS_EXTREMOS = parametros_features['umbral_minutos_extremos']
PESO_DESTINOS = parametros_features['peso_destinos']
PESO_SPRAY_RATIO = parametros_features['peso_spray_ratio']

print(f"\n⚙️ PARÁMETROS DE FEATURES CARGADOS:")
print(f"🔧 Peso minutos normal: {PESO_MINUTOS_NORMAL}")
print(f"🔧 Peso minutos extremos: {PESO_MINUTOS_EXTREMOS}")
print(f"🔧 Umbral minutos extremos: {UMBRAL_MINUTOS_EXTREMOS}")
print(f"🔧 Peso destinos: {PESO_DESTINOS}")
print(f"🔧 Peso spray ratio: {PESO_SPRAY_RATIO}")

# %% [markdown]
# # # 3. FUNCIÓN DE FEATURES (IDÉNTICA AL MODELO)
# 

# %%
def crear_features_contextualizadas_mejorada(row, stats_pais_dict):
    """
    Función idéntica a la del modelo - MANTENER CONSISTENCIA ABSOLUTA
    """
    pais = row['CODIGODEPAIS']
    llamadas = row['N_LLAMADAS']
    minutos = row['N_MINUTOS']
    destinos = row['N_DESTINOS']
    
    # Obtener contexto del país (si existe)
    if pais in stats_pais_dict:
        pais_stats = stats_pais_dict[pais]
        categoria = pais_stats['CATEGORIA']
        
        # Normalizar por el contexto del país
        llamadas_norm = min(llamadas / max(pais_stats['LLAMADAS_P95'], 1), 1.5)
        destinos_norm = min(destinos / max(pais_stats['DESTINOS_P95'], 1), 1.5)
        
        # Detección inteligente de minutos extremos
        minutos_p90 = pais_stats.get('MINUTOS_P90', pais_stats['MINUTOS_P95'] * 0.9)
        
        # Transformación adaptativa de minutos
        if minutos >= UMBRAL_MINUTOS_EXTREMOS:
            minutos_norm = min(minutos / max(minutos_p90, 1), 3.0)
            peso_minutos = PESO_MINUTOS_EXTREMOS
        else:
            minutos_norm = min(np.log1p(minutos) / np.log1p(max(minutos_p90, 1)), 1.2)
            peso_minutos = PESO_MINUTOS_NORMAL
            
    else:
        # País nuevo - SIEMPRE clasificar como 'Muy_Bajo'
        categoria = 'Muy_Bajo'
        llamadas_norm = min(llamadas / 10, 2.0)
        destinos_norm = min(destinos / 5, 2.0)
        
        if minutos >= UMBRAL_MINUTOS_EXTREMOS:
            minutos_norm = min(minutos / 50, 3.0)
            peso_minutos = PESO_MINUTOS_EXTREMOS * 1.2
        else:
            minutos_norm = min(np.log1p(minutos) / np.log1p(60), 1.2)
            peso_minutos = PESO_MINUTOS_NORMAL
    
    # Features principales (idénticas al modelo)
    features = {
        'llamadas_norm': llamadas_norm * 0.8,
        'destinos_norm': destinos_norm * PESO_DESTINOS,
        'minutos_norm': minutos_norm * peso_minutos,
        'diversidad_destinos': min(destinos / max(llamadas, 1), 1.0),
        'spray_ratio': min(destinos / max(llamadas, 1) * PESO_SPRAY_RATIO, 1.0) if destinos >= 5 else 0,
        'minutos_extremos': 1.0 if minutos >= UMBRAL_MINUTOS_EXTREMOS else 0.0,
        'minutos_sospechosos': min((minutos - 200) / 300, 1.0) if minutos > 200 else 0.0,
        'patron_spray_fuerte': 1.0 if (destinos >= 10 and llamadas >= 20) else 0.0,
        'patron_spray_medio': 0.5 if (destinos >= 6 and llamadas >= 12) else 0.0,
        'alta_diversidad': min(destinos / 12, 1) if destinos >= 5 else 0,
        'volumen_llamadas_alto': min((llamadas - 30) / 50, 1) if llamadas > 30 else 0,
        'volumen_destinos_alto': min((destinos - 10) / 20, 1) if destinos > 10 else 0,
        'llamadas_por_destino': min(llamadas / max(destinos, 1) / 5, 1),
        'eficiencia_destinos': min(destinos / max(llamadas * 0.5, 1), 1),
        'factor_pais_bajo': 1.5 if categoria in ['Muy_Bajo', 'Bajo'] else 1.0,
        'factor_pais_alto': 0.9 if categoria in ['Alto', 'Medio'] else 1.0
    }
    
    return features

# Función de predicción idéntica
def predecir_anomalia_mejorada(pais, linea, llamadas, minutos, destinos, modelo, scaler, umbral, stats_dict, contexto_historico=None):
    """
    Predicción idéntica al modelo original
    """
    # Crear row simulado
    row_data = {
        'CODIGODEPAIS': pais,
        'N_LLAMADAS': llamadas,
        'N_MINUTOS': minutos,
        'N_DESTINOS': destinos
    }
    
    # Crear features
    features = crear_features_contextualizadas_mejorada(row_data, stats_dict)
    
    # Normalizar
    features_scaled = scaler.transform_one(features)
    
    # Obtener score
    score = modelo.score_one(features_scaled)
    
    # Lógica de confirmación (idéntica al modelo)
    es_anomalia_base = score > umbral
    
    if es_anomalia_base:
        # Confirmar diferentes tipos de anomalías
        if minutos >= parametros_features['umbral_minutos_extremos']:
            es_anomalia_final = True
            razon = f"Minutos extremos ({minutos:.1f} min)"
        elif destinos >= 6 and llamadas >= 12:
            es_anomalia_final = True
            razon = "Patrón de spray calling confirmado"
        elif llamadas > 50 or destinos > 15:
            es_anomalia_final = True
            razon = "Volumen excepcionalmente alto"
        elif pais not in stats_dict or stats_dict.get(pais, {}).get('CATEGORIA') in ['Muy_Bajo', 'Bajo']:
            if destinos >= 4 and llamadas >= 8:
                es_anomalia_final = True
                razon = "Actividad sospechosa en país de bajo tráfico"
            else:
                es_anomalia_final = False
                razon = "Actividad baja en país de bajo tráfico"
        elif destinos < 3:
            es_anomalia_final = False
            razon = "Muy pocos destinos (<3)"
        elif destinos / max(llamadas, 1) < 0.15:
            es_anomalia_final = False
            razon = "Ratio destinos/llamadas muy bajo"
        elif llamadas < 5:
            es_anomalia_final = False
            razon = "Muy pocas llamadas (<5)"
        else:
            es_anomalia_final = False
            razon = "No cumple criterios de confirmación"
    else:
        es_anomalia_final = False
        razon = "Score bajo umbral"
    
    # Determinar contexto usando histórico
    if contexto_historico and pais in contexto_historico:
        tipo_contexto = contexto_historico[pais]
    elif pais in stats_dict:
        tipo_contexto = stats_dict[pais]['CATEGORIA']
    else:
        tipo_contexto = "Muy_Bajo"
    
    return {
        'score': score,
        'umbral': umbral,
        'es_anomalia': es_anomalia_final,
        'tipo_contexto': tipo_contexto,
        'razon_decision': razon,
        'features': features
    }

print("🔧 Funciones de predicción cargadas (idénticas al modelo)")

# %% [markdown]
# # # 19. CARGAR DATASET DE EVALUACIÓN CON ETIQUETAS DE FRAUDE

# %%
print(f"\n📂 Cargando dataset de evaluación...")

# Verificar que el archivo existe
if not os.path.exists(EVALUATION_CSV_PATH):
    print(f"❌ Error: Archivo de evaluación no encontrado: {EVALUATION_CSV_PATH}")
    print(f"📝 Por favor, asegúrate de que el archivo exista y contenga las columnas:")
    print(f"   - FECHA, CODIGODEPAIS, LINEA, N_LLAMADAS, N_MINUTOS, N_DESTINOS, FRAUDE")
    print(f"   - FRAUDE debe ser 1 (fraudulento) o 0 (normal)")
    exit()

# Cargar dataset
df_evaluacion = pd.read_csv(EVALUATION_CSV_PATH)

# Convertir fecha a datetime si existe
if 'FECHA' in df_evaluacion.columns:
    df_evaluacion['FECHA'] = pd.to_datetime(df_evaluacion['FECHA'], format='%d/%m/%Y', errors='coerce')

print(f"✅ Dataset cargado - Shape: {df_evaluacion.shape}")

# Verificar columnas requeridas
columnas_requeridas = ['CODIGODEPAIS', 'LINEA', 'N_LLAMADAS', 'N_MINUTOS', 'N_DESTINOS', 'FRAUDE']
columnas_faltantes = [col for col in columnas_requeridas if col not in df_evaluacion.columns]

if columnas_faltantes:
    print(f"❌ Error: Columnas faltantes: {columnas_faltantes}")
    print(f"📋 Columnas disponibles: {list(df_evaluacion.columns)}")
    exit()

# Verificar valores de FRAUDE
valores_fraude = df_evaluacion['FRAUDE'].unique()
if not all(v in [0, 1] for v in valores_fraude):
    print(f"❌ Error: FRAUDE debe contener solo valores 0 o 1. Valores encontrados: {valores_fraude}")
    exit()

print(f"🔍 ANÁLISIS DEL DATASET DE EVALUACIÓN:")
print(f"📊 Total de registros: {len(df_evaluacion)}")
print(f"🚨 Casos de fraude: {df_evaluacion['FRAUDE'].sum()} ({df_evaluacion['FRAUDE'].mean()*100:.2f}%)")
print(f"✅ Casos normales: {(df_evaluacion['FRAUDE'] == 0).sum()} ({(df_evaluacion['FRAUDE'] == 0).mean()*100:.2f}%)")
print(f"🌍 Países únicos: {df_evaluacion['CODIGODEPAIS'].nunique()}")
print(f"📞 Líneas únicas: {df_evaluacion['LINEA'].nunique()}")

# Mostrar estadísticas por clase
print(f"\n📊 ESTADÍSTICAS POR CLASE:")
print(f"CASOS NORMALES (FRAUDE = 0):")
normales = df_evaluacion[df_evaluacion['FRAUDE'] == 0]
print(f"  📞 Llamadas - Min: {normales['N_LLAMADAS'].min()}, Max: {normales['N_LLAMADAS'].max()}, Media: {normales['N_LLAMADAS'].mean():.1f}")
print(f"  ⏱️ Minutos - Min: {normales['N_MINUTOS'].min()}, Max: {normales['N_MINUTOS'].max()}, Media: {normales['N_MINUTOS'].mean():.1f}")
print(f"  🎯 Destinos - Min: {normales['N_DESTINOS'].min()}, Max: {normales['N_DESTINOS'].max()}, Media: {normales['N_DESTINOS'].mean():.1f}")

if df_evaluacion['FRAUDE'].sum() > 0:
    print(f"\nCASOS DE FRAUDE (FRAUDE = 1):")
    fraudes = df_evaluacion[df_evaluacion['FRAUDE'] == 1]
    print(f"  📞 Llamadas - Min: {fraudes['N_LLAMADAS'].min()}, Max: {fraudes['N_LLAMADAS'].max()}, Media: {fraudes['N_LLAMADAS'].mean():.1f}")
    print(f"  ⏱️ Minutos - Min: {fraudes['N_MINUTOS'].min()}, Max: {fraudes['N_MINUTOS'].max()}, Media: {fraudes['N_MINUTOS'].mean():.1f}")
    print(f"  🎯 Destinos - Min: {fraudes['N_DESTINOS'].min()}, Max: {fraudes['N_DESTINOS'].max()}, Media: {fraudes['N_DESTINOS'].mean():.1f}")


# %% [markdown]
# # # 20. REALIZAR PREDICCIONES EN DATASET DE EVALUACIÓN

# %%
print(f"\n🔮 Realizando predicciones en dataset de evaluación...")

predicciones = []
scores = []

for contador, (idx, row) in enumerate(df_evaluacion.iterrows()):
    if contador % 1000 == 0:
        print(f"   Procesando: {contador}/{len(df_evaluacion)} registros")
    
    # Realizar predicción
    resultado = predecir_anomalia_mejorada(
        pais=row['CODIGODEPAIS'],
        linea=row['LINEA'],
        llamadas=row['N_LLAMADAS'],
        minutos=row['N_MINUTOS'],
        destinos=row['N_DESTINOS'],
        modelo=modelo_cargado,
        scaler=scaler_cargado,
        umbral=umbral_global,
        stats_dict=stats_dict,
        contexto_historico=contexto_historico
    )
    
    # Guardar predicción (1 si es anomalía, 0 si es normal)
    predicciones.append(1 if resultado['es_anomalia'] else 0)
    scores.append(resultado['score'])

# Agregar predicciones al dataframe
df_evaluacion['PREDICCION'] = predicciones
df_evaluacion['SCORE_ANOMALIA'] = scores

print(f"✅ Predicciones completadas")

# %% [markdown]
# # # 21. CALCULAR MÉTRICAS CON SISTEMA DE MEMORIA DE FRAUDE

# %%
print(f"\n📊 CALCULANDO MÉTRICAS CON MEMORIA DE FRAUDE...")

# Clase para memoria de fraude
class SistemaMemoriaFraude:
    def __init__(self):
        self.memoria_fraude = {}  # {linea: fecha_primera_deteccion}
        self.detecciones_por_memoria = 0
    
    def registrar_deteccion_fraude(self, linea, fecha):
        if linea not in self.memoria_fraude:
            self.memoria_fraude[linea] = pd.to_datetime(fecha)
    
    def es_fraude_por_memoria(self, linea, fecha_actual):
        if linea not in self.memoria_fraude:
            return False
        return pd.to_datetime(fecha_actual) >= self.memoria_fraude[linea]
    
    def aplicar_memoria_fraude(self, linea, fecha_actual, prediccion_original):
        if self.es_fraude_por_memoria(linea, fecha_actual) and prediccion_original == 0:
            self.detecciones_por_memoria += 1
            return 1
        return prediccion_original

# Verificar/crear columna de fecha
if 'FECHA' not in df_evaluacion.columns:
    print("⚠️ Agregando fechas simuladas...")
    fechas_base = pd.date_range(start='2024-01-01', periods=len(df_evaluacion), freq='D')
    df_evaluacion['FECHA'] = np.random.choice(fechas_base, len(df_evaluacion))

# Ordenar por fecha para procesamiento secuencial
df_evaluacion['FECHA'] = pd.to_datetime(df_evaluacion['FECHA'])
df_evaluacion = df_evaluacion.sort_values(['FECHA', 'LINEA']).reset_index(drop=True)

# Inicializar sistema de memoria
sistema_memoria = SistemaMemoriaFraude()

# Procesar con memoria
predicciones_con_memoria = []
for idx, row in df_evaluacion.iterrows():
    # Predicción original (ya calculada)
    prediccion_original = predicciones[idx]
    
    # Aplicar memoria
    prediccion_con_memoria = sistema_memoria.aplicar_memoria_fraude(
        row['LINEA'], row['FECHA'], prediccion_original
    )
    predicciones_con_memoria.append(prediccion_con_memoria)
    
    # Si detectamos fraude, registrar en memoria
    if prediccion_con_memoria == 1:
        sistema_memoria.registrar_deteccion_fraude(row['LINEA'], row['FECHA'])

# Agregar columnas
df_evaluacion['PREDICCION_ORIGINAL'] = predicciones
df_evaluacion['PREDICCION_CON_MEMORIA'] = predicciones_con_memoria

# MÉTRICAS ORIGINALES
y_true = df_evaluacion['FRAUDE'].values
y_pred_original = df_evaluacion['PREDICCION_ORIGINAL'].values

accuracy = accuracy_score(y_true, y_pred_original)
precision = precision_score(y_true, y_pred_original, zero_division=0)
recall = recall_score(y_true, y_pred_original, zero_division=0)
f1 = f1_score(y_true, y_pred_original, zero_division=0)

print(f"🎯 MÉTRICAS ORIGINALES:")
print(f"📈 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"🔍 Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"⚖️ F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# MÉTRICAS CON MEMORIA
y_pred_memoria = df_evaluacion['PREDICCION_CON_MEMORIA'].values

accuracy_mem = accuracy_score(y_true, y_pred_memoria)
precision_mem = precision_score(y_true, y_pred_memoria, zero_division=0)
recall_mem = recall_score(y_true, y_pred_memoria, zero_division=0)
f1_mem = f1_score(y_true, y_pred_memoria, zero_division=0)

print(f"\n🧠 MÉTRICAS CON MEMORIA:")
print(f"📈 Accuracy: {accuracy_mem:.4f} ({accuracy_mem*100:.2f}%)")
print(f"🎯 Precision: {precision_mem:.4f} ({precision_mem*100:.2f}%)")
print(f"🔍 Recall: {recall_mem:.4f} ({recall_mem*100:.2f}%)")
print(f"⚖️ F1-Score: {f1_mem:.4f} ({f1_mem*100:.2f}%)")

print(f"\n📈 MEJORAS:")
print(f"Accuracy: {accuracy_mem - accuracy:+.4f}")
print(f"Precision: {precision_mem - precision:+.4f}")
print(f"Recall: {recall_mem - recall:+.4f}")
print(f"F1-Score: {f1_mem - f1:+.4f}")

# Matriz de confusión original
cm = confusion_matrix(y_true, y_pred_original)
tn, fp, fn, tp = cm.ravel()

print(f"\n📋 MATRIZ DE CONFUSIÓN ORIGINAL:")
print(f"┌─────────────────┬─────────────────┐")
print(f"│     REAL \\ PRED │   Normal   │ Anomalía │")
print(f"├─────────────────┼─────────────────┤")
print(f"│      Normal     │   {tn:6d}   │  {fp:6d}  │")
print(f"│     Fraude      │   {fn:6d}   │  {tp:6d}  │")
print(f"└─────────────────┴─────────────────┘")

# Matriz de confusión con memoria
cm_mem = confusion_matrix(y_true, y_pred_memoria)
tn_mem, fp_mem, fn_mem, tp_mem = cm_mem.ravel()

print(f"\n📋 MATRIZ DE CONFUSIÓN CON MEMORIA:")
print(f"┌─────────────────┬─────────────────┐")
print(f"│     REAL \\ PRED │   Normal   │ Anomalía │")
print(f"├─────────────────┼─────────────────┤")
print(f"│      Normal     │   {tn_mem:6d}   │  {fp_mem:6d}  │")
print(f"│     Fraude      │   {fn_mem:6d}   │  {tp_mem:6d}  │")
print(f"└─────────────────┴─────────────────┘")

print(f"\n🧠 ESTADÍSTICAS DE MEMORIA:")
print(f"📞 Líneas en memoria: {len(sistema_memoria.memoria_fraude)}")
print(f"🔄 Detecciones por memoria: {sistema_memoria.detecciones_por_memoria}")
print(f"📊 Fraudes adicionales detectados: {tp_mem - tp}")

# %%
# Guardar resultados con memoria
resultados_memoria_path = os.path.join(RESULTADO, "resultados_evaluacion_con_memoria.csv")
df_evaluacion.to_csv(resultados_memoria_path, index=False)
print(f"📄 Resultados con memoria: {resultados_memoria_path}")


