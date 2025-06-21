from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
import asyncio
import pickle
import shutil
import logging

from ..models.schemas import (
    RegistroTelefonico,
    ResultadoScoring,
    EstadisticasModelo,
    ConfiguracionUmbral,
    EstadoSistema,
    TipoAnomalia,
)
from ..models import ml
from ..database.oracle import (
    registrar_lote_procesamiento,
    guardar_anomalias_oracle,
    get_oracle_connection,
)
from ..database import oracle

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "API de Detección de Fraude Telefónico",
        "version": "1.0",
        "documentacion": "/docs",
        "estado": "activo" if ml.modelo else "modelo no cargado"
    }

@router.post("/scoring/individual", response_model=ResultadoScoring, tags=["Scoring"])
async def scoring_individual(registro: RegistroTelefonico):
    """
    Realiza scoring de un registro individual
    """
    if not ml.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Preparar datos
        row_dict = {
            'CODIGODEPAIS': registro.codigodepais,
            'N_LLAMADAS': registro.n_llamadas,
            'N_MINUTOS': registro.n_minutos,
            'N_DESTINOS': registro.n_destinos
        }

        # Realizar predicción
        resultado = ml.predecir_anomalia_individual(row_dict)

        # Construir respuesta
        return ResultadoScoring(
            fecha=registro.fecha,
            codigodepais=registro.codigodepais,
            linea=registro.linea,
            n_llamadas=registro.n_llamadas,
            n_minutos=registro.n_minutos,
            n_destinos=registro.n_destinos,
            score_anomalia=round(resultado['score'], 4),
            umbral=round(resultado['umbral'], 4),
            es_anomalia=resultado['es_anomalia'],
            tipo_anomalia=resultado['tipo_anomalia'],
            tipo_contexto=resultado['tipo_contexto'],
            razon_decision=resultado['razon_decision'],
            timestamp_procesamiento=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error en scoring individual: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scoring/batch", tags=["Scoring"])
async def scoring_batch(file: UploadFile = File(...)):
    """
    Realiza scoring de un archivo CSV con múltiples registros
    y guarda anomalías en Oracle
    """
    if not ml.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    lote_id = None
    temp_file = None

    try:
        # Guardar archivo temporal
        temp_file = str(ml.TEMP_PATH / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")

        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # Leer CSV
        df = pd.read_csv(temp_file)

        # Convertir CODIGODEPAIS a int si es necesario
        if 'CODIGODEPAIS' in df.columns:
            df['CODIGODEPAIS'] = pd.to_numeric(df['CODIGODEPAIS'], errors='coerce').astype('Int64')

        # Validar columnas
        columnas_requeridas = ['FECHA', 'CODIGODEPAIS', 'LINEA', 'N_LLAMADAS', 'N_MINUTOS', 'N_DESTINOS']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

        if columnas_faltantes:
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {columnas_faltantes}")

        # Procesar registros
        resultados = []
        anomalias_detectadas = 0

        for idx, row in df.iterrows():
            # Saltar filas con CODIGODEPAIS nulo
            if pd.isna(row['CODIGODEPAIS']):
                continue

            row_dict = {
                'CODIGODEPAIS': int(row['CODIGODEPAIS']),
                'N_LLAMADAS': row['N_LLAMADAS'],
                'N_MINUTOS': row['N_MINUTOS'],
                'N_DESTINOS': row['N_DESTINOS']
            }

            resultado = ml.predecir_anomalia_individual(row_dict)

            resultado_completo = {
                'FECHA': row['FECHA'],
                'CODIGODEPAIS': int(row['CODIGODEPAIS']),
                'LINEA': row['LINEA'],
                'N_LLAMADAS': row['N_LLAMADAS'],
                'N_MINUTOS': row['N_MINUTOS'],
                'N_DESTINOS': row['N_DESTINOS'],
                'score_anomalia': round(resultado['score'], 4),
                'umbral': round(resultado['umbral'], 4),
                'es_anomalia': resultado['es_anomalia'],
                'tipo_anomalia': resultado['tipo_anomalia'].value,
                'tipo_contexto': resultado['tipo_contexto'],
                'razon_decision': resultado['razon_decision'],
                'timestamp_procesamiento': datetime.now()
            }

            resultados.append(resultado_completo)
            if resultado['es_anomalia']:
                anomalias_detectadas += 1

        # Guardar resultados en CSV
        df_resultados = pd.DataFrame(resultados)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo_salida = str(ml.OUTPUT_PATH / f"resultados_batch_{timestamp}.csv")
        df_resultados.to_csv(archivo_salida, index=False)

        # NUEVO: Guardar en Oracle si está disponible
        if oracle.oracle_pool:
            try:
                # Registrar lote
                lote_id = registrar_lote_procesamiento(
                    archivo_entrada=file.filename,
                    total_registros=len(resultados),
                    total_anomalias=anomalias_detectadas,
                    tasa_anomalias=round(anomalias_detectadas/len(resultados)*100, 2) if len(resultados) > 0 else 0,
                    archivo_salida=os.path.basename(archivo_salida)
                )

                # Guardar solo las anomalías
                df_anomalias = df_resultados[df_resultados['es_anomalia'] == True]
                if not df_anomalias.empty:
                    guardar_anomalias_oracle(df_anomalias, file.filename, str(lote_id))

                logger.info(f"Datos guardados en Oracle - Lote ID: {lote_id}")

            except Exception as e:
                logger.error(f"Error guardando en Oracle: {e}")
                # No fallar el proceso si Oracle falla

        # Limpiar archivo temporal
        os.remove(temp_file)

        # Respuesta
        return {
            "mensaje": "Procesamiento completado",
            "registros_procesados": len(resultados),
            "anomalias_detectadas": anomalias_detectadas,
            "tasa_anomalias": round(anomalias_detectadas/len(resultados)*100, 2) if len(resultados) > 0 else 0,
            "archivo_resultados": os.path.basename(archivo_salida),
            "lote_id": lote_id,
            "timestamp": datetime.now(),
            "guardado_oracle": oracle.oracle_pool is not None
        }

    except Exception as e:
        logger.error(f"Error en scoring batch: {str(e)}")
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

        # Registrar error en Oracle si es posible
        if oracle.oracle_pool and lote_id is None:
            try:
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO LOTES_PROCESAMIENTO 
                        (TIMESTAMP_INICIO, ARCHIVO_ENTRADA, ESTADO, MENSAJE_ERROR)
                        VALUES (:1, :2, :3, :4)
                    """, [datetime.now(), file.filename, "ERROR", str(e)])
                    conn.commit()
            except:
                pass

        raise HTTPException(status_code=500, detail=str(e))


# # ## NUEVO ENDPOINT CONSULTAS ORACLE

# In[11]:


@router.get("/alertas/consultar", tags=["Alertas Oracle"])
async def consultar_alertas(
    fecha_inicio: Optional[str] = Query(None, description="Fecha inicio YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha fin YYYY-MM-DD"),
    codigo_pais: Optional[int] = Query(None, description="Código de país"),
    tipo_anomalia: Optional[str] = Query(None, description="Tipo de anomalía"),
    limite: int = Query(100, description="Número máximo de registros")
):
    """
    Consulta las alertas guardadas en Oracle con filtros opcionales
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            # Construir query dinámicamente
            query = """
                SELECT ID_ALERTA, FECHA_PROCESAMIENTO, FECHA_REGISTRO, CODIGO_PAIS, 
                       LINEA, N_LLAMADAS, N_MINUTOS, N_DESTINOS, SCORE_ANOMALIA,
                       UMBRAL, TIPO_ANOMALIA, TIPO_CONTEXTO, RAZON_DECISION
                FROM ALERTAS_FRAUDE
                WHERE 1=1
            """
            params = []

            if fecha_inicio:
                query += " AND FECHA_PROCESAMIENTO >= TO_DATE(:fecha_inicio, 'YYYY-MM-DD')"
                params.append(fecha_inicio)

            if fecha_fin:
                query += " AND FECHA_PROCESAMIENTO <= TO_DATE(:fecha_fin, 'YYYY-MM-DD') + 1"
                params.append(fecha_fin)

            if codigo_pais:
                query += " AND CODIGO_PAIS = :codigo_pais"
                params.append(codigo_pais)

            if tipo_anomalia:
                query += " AND TIPO_ANOMALIA = :tipo_anomalia"
                params.append(tipo_anomalia)

            query += " ORDER BY FECHA_PROCESAMIENTO DESC"
            query += f" FETCH FIRST {limite} ROWS ONLY"

            cursor.execute(query, params)

            # Obtener nombres de columnas
            columnas = [col[0] for col in cursor.description]

            # Convertir a lista de diccionarios
            alertas = []
            for row in cursor:
                alerta = dict(zip(columnas, row))
                # Convertir datetime a string
                if alerta['FECHA_PROCESAMIENTO']:
                    alerta['FECHA_PROCESAMIENTO'] = alerta['FECHA_PROCESAMIENTO'].isoformat()
                alertas.append(alerta)

            return {
                "total": len(alertas),
                "alertas": alertas
            }

    except Exception as e:
        logger.error(f"Error consultando alertas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alertas/estadisticas", tags=["Alertas Oracle"])
async def estadisticas_alertas(
    fecha_inicio: str = Query(..., description="Fecha inicio YYYY-MM-DD"),
    fecha_fin: str = Query(..., description="Fecha fin YYYY-MM-DD")
):
    """
    Obtiene estadísticas de las alertas en un rango de fechas
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            # Estadísticas generales
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alertas,
                    COUNT(DISTINCT CODIGO_PAIS) as paises_afectados,
                    COUNT(DISTINCT LINEA) as lineas_afectadas,
                    AVG(SCORE_ANOMALIA) as score_promedio,
                    MAX(SCORE_ANOMALIA) as score_maximo
                FROM ALERTAS_FRAUDE
                WHERE FECHA_PROCESAMIENTO >= TO_DATE(:1, 'YYYY-MM-DD')
                  AND FECHA_PROCESAMIENTO <= TO_DATE(:2, 'YYYY-MM-DD') + 1
            """, [fecha_inicio, fecha_fin])

            stats_generales = dict(zip(
                ['total_alertas', 'paises_afectados', 'lineas_afectadas', 
                 'score_promedio', 'score_maximo'],
                cursor.fetchone()
            ))

            # Distribución por tipo
            cursor.execute("""
                SELECT TIPO_ANOMALIA, COUNT(*) as cantidad
                FROM ALERTAS_FRAUDE
                WHERE FECHA_PROCESAMIENTO >= TO_DATE(:1, 'YYYY-MM-DD')
                  AND FECHA_PROCESAMIENTO <= TO_DATE(:2, 'YYYY-MM-DD') + 1
                GROUP BY TIPO_ANOMALIA
                ORDER BY cantidad DESC
            """, [fecha_inicio, fecha_fin])

            distribucion_tipos = [
                {"tipo": row[0], "cantidad": row[1]}
                for row in cursor.fetchall()
            ]

            # Top países
            cursor.execute("""
                SELECT CODIGO_PAIS, COUNT(*) as cantidad
                FROM ALERTAS_FRAUDE
                WHERE FECHA_PROCESAMIENTO >= TO_DATE(:1, 'YYYY-MM-DD')
                  AND FECHA_PROCESAMIENTO <= TO_DATE(:2, 'YYYY-MM-DD') + 1
                GROUP BY CODIGO_PAIS
                ORDER BY cantidad DESC
                FETCH FIRST 10 ROWS ONLY
            """, [fecha_inicio, fecha_fin])

            top_paises = [
                {"codigo_pais": row[0], "cantidad": row[1]}
                for row in cursor.fetchall()
            ]

            # Tendencia diaria
            cursor.execute("""
                SELECT 
                    TO_CHAR(FECHA_PROCESAMIENTO, 'YYYY-MM-DD') as fecha,
                    COUNT(*) as cantidad
                FROM ALERTAS_FRAUDE
                WHERE FECHA_PROCESAMIENTO >= TO_DATE(:1, 'YYYY-MM-DD')
                  AND FECHA_PROCESAMIENTO <= TO_DATE(:2, 'YYYY-MM-DD') + 1
                GROUP BY TO_CHAR(FECHA_PROCESAMIENTO, 'YYYY-MM-DD')
                ORDER BY fecha
            """, [fecha_inicio, fecha_fin])

            tendencia_diaria = [
                {"fecha": row[0], "cantidad": row[1]}
                for row in cursor.fetchall()
            ]

            return {
                "periodo": {
                    "fecha_inicio": fecha_inicio,
                    "fecha_fin": fecha_fin
                },
                "estadisticas_generales": stats_generales,
                "distribucion_tipos": distribucion_tipos,
                "top_paises": top_paises,
                "tendencia_diaria": tendencia_diaria
            }

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lotes/historial", tags=["Alertas Oracle"])
async def historial_lotes(limite: int = Query(50, description="Número de lotes a mostrar")):
    """
    Obtiene el historial de lotes procesados
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT ID_LOTE, TIMESTAMP_INICIO, TIMESTAMP_FIN, ARCHIVO_ENTRADA,
                       TOTAL_REGISTROS, TOTAL_ANOMALIAS, TASA_ANOMALIAS,
                       ARCHIVO_SALIDA, ESTADO
                FROM LOTES_PROCESAMIENTO
                ORDER BY TIMESTAMP_INICIO DESC
                FETCH FIRST :limite ROWS ONLY
            """, [limite])

            columnas = [col[0] for col in cursor.description]
            lotes = []

            for row in cursor:
                lote = dict(zip(columnas, row))
                # Convertir timestamps
                if lote['TIMESTAMP_INICIO']:
                    lote['TIMESTAMP_INICIO'] = lote['TIMESTAMP_INICIO'].isoformat()
                if lote['TIMESTAMP_FIN']:
                    lote['TIMESTAMP_FIN'] = lote['TIMESTAMP_FIN'].isoformat()
                lotes.append(lote)

            return {
                "total": len(lotes),
                "lotes": lotes
            }

    except Exception as e:
        logger.error(f"Error obteniendo historial de lotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# # ##FUNCIONES PARA APRENDISAJE INCREMENTAL / ORACLE

# In[12]:


def determinar_categoria_pais(stats):
    """Determina la categoría de un país basado en sus estadísticas"""
    # Basado en el promedio de llamadas
    llamadas_mean = stats.get('LLAMADAS_MEAN', 0)

    if llamadas_mean >= 50:
        return 'Alto'
    elif llamadas_mean >= 20:
        return 'Medio'
    elif llamadas_mean >= 10:
        return 'Bajo'
    else:
        return 'Muy_Bajo'

def calcular_umbral_adaptativo(modelo, scaler, df_muestra, stats_dict):
    """Calcula un umbral adaptativo basado en los scores actuales"""
    scores = []

    # Tomar una muestra aleatoria
    muestra = df_muestra.sample(min(1000, len(df_muestra)))

    for _, row in muestra.iterrows():
        row_dict = {
            'CODIGODEPAIS': int(row['CODIGODEPAIS']) if not pd.isna(row['CODIGODEPAIS']) else 0,
            'N_LLAMADAS': row['N_LLAMADAS'],
            'N_MINUTOS': row['N_MINUTOS'],
            'N_DESTINOS': row['N_DESTINOS']
        }

        features = crear_features_contextualizadas_mejorada(row_dict, ml.stats_dict)
        features_scaled = scaler.transform_one(features)
        score = modelo.score_one(features_scaled)
        scores.append(score)

    # Nuevo umbral = percentil 95 de los scores
    return np.percentile(scores, 95)

def registrar_inicio_actualizacion_diaria(archivo_path, total_registros, estado_inicial):
    """Registra el inicio de la actualización diaria en Oracle"""
    with get_oracle_connection() as conn:
        cursor = conn.cursor()

        # Contar países únicos en el archivo
        try:
            df_temp = pd.read_csv(archivo_path)
            paises_en_archivo = df_temp['CODIGODEPAIS'].nunique()
        except:
            paises_en_archivo = 0

        id_var = cursor.var(oracledb.NUMBER)

        cursor.execute("""
            INSERT INTO APRENDIZAJE_INCREMENTAL (
                FECHA_DATOS, ARCHIVO_ENTRADA, TOTAL_REGISTROS,
                PAISES_TOTALES, UMBRAL_ANTERIOR, N_TREES_ANTERIOR,
                PAISES_CONOCIDOS_ANTERIOR, ESTADO
            ) VALUES (
                TRUNC(SYSDATE), :1, :2, :3, :4, :5, :6, 'EN_PROCESO'
            ) RETURNING ID_APRENDIZAJE INTO :id
        """, [
            os.path.basename(archivo_path),
            total_registros,
            paises_en_archivo,
            estado_inicial['umbral'],
            estado_inicial['n_trees'],
            estado_inicial['paises_conocidos'],
            id_var
        ])

        conn.commit()
        return id_var.getvalue()[0]

def actualizar_fin_actualizacion_diaria(id_actualizacion, **kwargs):
    """Actualiza el registro al finalizar la actualización diaria"""
    with get_oracle_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE ACTUALIZACION_DIARIA_UMBRAL SET
                REGISTROS_PROCESADOS = :1,
                UMBRAL_NUEVO = :2,
                TIEMPO_PROCESAMIENTO_SEG = :3,
                ESTADO = :4,
                ARCHIVO_BACKUP_CONFIG = :5,
                OBSERVACIONES = :6
            WHERE ID_ACTUALIZACION = :7
        """, [
            kwargs.get('registros_analizados', 0),
            kwargs['umbral_nuevo'],
            kwargs['tiempo_procesamiento'],
            kwargs['estado'],
            os.path.basename(kwargs['backup_path']),
            f"Cambio: {kwargs.get('razon_cambio', 'N/A')}, Precision: {kwargs.get('metricas_performance', {}).get('precision', 0):.2f}",
            id_actualizacion
        ])
        
        conn.commit()

# # ## 6. ENDPOINTS DE APRENDIZAJE INCREMENTAL
# 

# In[13]:


@router.post("/modelo/actualizacion-diaria", tags=["Modelo"])
async def actualizacion_diaria(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Realiza la actualización diaria del umbral basándose en los datos del día
    NO modifica el modelo River, solo recalcula el umbral óptimo
    """
    if not ml.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Guardar archivo temporal
        temp_file = str(ml.TEMP_PATH / f"diario_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validar archivo
        df = pd.read_csv(temp_file)
        
        # Verificar que sean datos de un solo día
        if 'FECHA' in df.columns:
            fechas_unicas = df['FECHA'].nunique()
            if fechas_unicas > 1:
                logger.warning(f"El archivo contiene datos de {fechas_unicas} días diferentes")
        
        # Agregar tarea en background
        background_tasks.add_task(procesar_actualizacion_diaria, temp_file, df)
        
        return {
            "mensaje": "Actualización diaria del umbral iniciada",
            "registros_a_analizar": len(df),
            "umbral_actual": ml.umbral_global,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error iniciando actualización diaria: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

def procesar_actualizacion_diaria(archivo_path: str, df: pd.DataFrame):
    """
    Procesa la actualización diaria del modelo River HalfSpaceTree
    NO es aprendizaje incremental - es recálculo completo del umbral
    """
    
    # Variables para tracking
    tiempo_inicio = datetime.now()
    id_actualizacion = None
    estado_inicial = {
        'umbral': ml.umbral_global,
        'n_trees': ml.config.get('n_trees', 0),
        'paises_conocidos': len(ml.stats_dict)
    }
    
    try:
        logger.info(f"Iniciando actualización diaria con {len(df)} registros del día")
        
        # Registrar inicio en Oracle
        if oracle.oracle_pool:
            id_actualizacion = registrar_inicio_actualizacion_diaria(
                archivo_path, len(df), estado_inicial
            )
        
        # 1. CALCULAR SCORES PARA TODOS LOS REGISTROS DEL DÍA
        scores_dia = []
        scores_por_tipo = {
            'normales': [],
            'minutos_extremos': [],
            'spray_calling': [],
            'volumen_alto': [],
            'pais_bajo': []
        }
        
        for idx, row in df.iterrows():
            row_dict = {
                'CODIGODEPAIS': int(row['CODIGODEPAIS']) if not pd.isna(row['CODIGODEPAIS']) else 0,
                'N_LLAMADAS': row['N_LLAMADAS'],
                'N_MINUTOS': row['N_MINUTOS'],
                'N_DESTINOS': row['N_DESTINOS']
            }
            
            # Crear features
            features = crear_features_contextualizadas_mejorada(row_dict, ml.stats_dict)
            features_scaled = scaler.transform_one(features)
            
            # Obtener score del modelo River
            score = modelo.score_one(features_scaled)
            scores_dia.append(score)
            
            # Clasificar por tipo para análisis
            pais = row_dict['CODIGODEPAIS']
            minutos = row_dict['N_MINUTOS']
            llamadas = row_dict['N_LLAMADAS']
            destinos = row_dict['N_DESTINOS']
            
            if minutos >= ml.parametros_features['umbral_minutos_extremos']:
                scores_por_tipo['minutos_extremos'].append(score)
            elif destinos >= 6 and llamadas >= 12:
                scores_por_tipo['spray_calling'].append(score)
            elif llamadas > 50 or destinos > 15:
                scores_por_tipo['volumen_alto'].append(score)
            elif pais not in ml.stats_dict or ml.stats_dict.get(pais, {}).get('CATEGORIA') in ['Muy_Bajo', 'Bajo']:
                scores_por_tipo['pais_bajo'].append(score)
            else:
                scores_por_tipo['normales'].append(score)
        
        # 2. ANÁLISIS ESTADÍSTICO DE SCORES
        analisis_scores = {
            'media': np.mean(scores_dia),
            'mediana': np.median(scores_dia),
            'std': np.std(scores_dia),
            'p90': np.percentile(scores_dia, 90),
            'p95': np.percentile(scores_dia, 95),
            'p99': np.percentile(scores_dia, 99),
            'min': np.min(scores_dia),
            'max': np.max(scores_dia)
        }
        
        # 3. CALCULAR NUEVO UMBRAL ADAPTATIVO
        # Método 1: Basado en distribución estadística
        umbral_estadistico = analisis_scores['p95']
        
        # Método 2: Separación entre normales y anómalos
        if len(scores_por_tipo['normales']) > 0:
            umbral_separacion = np.percentile(scores_por_tipo['normales'], 99)
        else:
            umbral_separacion = umbral_estadistico
        
        # Método 3: Considerar tipos específicos
        umbrales_por_tipo = {}
        for tipo, scores in scores_por_tipo.items():
            if len(scores) > 0:
                if tipo == 'normales':
                    umbrales_por_tipo[tipo] = np.percentile(scores, 98)
                else:
                    umbrales_por_tipo[tipo] = np.percentile(scores, 50)
        
        # 4. DECISIÓN FINAL DEL UMBRAL
        # Tomar el mínimo entre estadístico y separación para ser conservador
        umbral_candidato = min(umbral_estadistico, umbral_separacion)
        
        # Aplicar límites de cambio (máximo 10% de cambio por día)
        cambio_maximo = ml.umbral_global * 0.1
        
        if umbral_candidato > ml.umbral_global + cambio_maximo:
            umbral_nuevo = ml.umbral_global + cambio_maximo
            razon_cambio = "Incremento limitado al 10%"
        elif umbral_candidato < ml.umbral_global - cambio_maximo:
            umbral_nuevo = ml.umbral_global - cambio_maximo
            razon_cambio = "Decremento limitado al 10%"
        else:
            umbral_nuevo = umbral_candidato
            razon_cambio = "Ajuste dentro del rango permitido"
        
        # 5. VALIDAR NUEVO UMBRAL CON DATOS DEL DÍA
        falsos_positivos = 0
        falsos_negativos = 0
        verdaderos_positivos = 0
        verdaderos_negativos = 0
        
        for idx, score in enumerate(scores_dia):
            row = df.iloc[idx]
            
            # Predicción con nuevo umbral
            es_anomalia_nuevo = score > umbral_nuevo
            
            # Verificar si debería ser anomalía según reglas de negocio
            es_anomalia_real = False
            if row['N_MINUTOS'] >= ml.parametros_features['umbral_minutos_extremos']:
                es_anomalia_real = True
            elif row['N_DESTINOS'] >= 10 and row['N_LLAMADAS'] >= 20:
                es_anomalia_real = True
            elif row['N_LLAMADAS'] > 100:
                es_anomalia_real = True
            
            # Calcular métricas
            if es_anomalia_nuevo and es_anomalia_real:
                verdaderos_positivos += 1
            elif es_anomalia_nuevo and not es_anomalia_real:
                falsos_positivos += 1
            elif not es_anomalia_nuevo and es_anomalia_real:
                falsos_negativos += 1
            else:
                verdaderos_negativos += 1
        
        # Calcular métricas de performance
        precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
        recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 6. ACTUALIZAR ESTADÍSTICAS POR PAÍS (sin cambiar el modelo)
        estadisticas_dia = {}
        for pais in df['CODIGODEPAIS'].unique():
            if pd.isna(pais):
                continue
                
            pais = int(pais)
            datos_pais = df[df['CODIGODEPAIS'] == pais]
            
            estadisticas_dia[pais] = {
                'registros': len(datos_pais),
                'llamadas_mean': datos_pais['N_LLAMADAS'].mean(),
                'minutos_mean': datos_pais['N_MINUTOS'].mean(),
                'destinos_mean': datos_pais['N_DESTINOS'].mean(),
                'llamadas_max': datos_pais['N_LLAMADAS'].max(),
                'minutos_max': datos_pais['N_MINUTOS'].max(),
                'destinos_max': datos_pais['N_DESTINOS'].max()
            }
        
        # 7. ACTUALIZAR CONFIGURACIÓN
        ml.umbral_global = umbral_nuevo
        ml.config['umbral_global'] = ml.umbral_global
        ml.config['fecha_ultima_actualizacion'] = datetime.now().isoformat()
        ml.config['ultima_actualizacion_diaria'] = {
            'fecha': datetime.now().isoformat(),
            'registros_analizados': len(df),
            'umbral_anterior': estado_inicial['umbral'],
            'umbral_nuevo': umbral_nuevo,
            'cambio_porcentual': ((umbral_nuevo - estado_inicial['umbral']) / estado_inicial['umbral'] * 100),
            'razon_cambio': razon_cambio,
            'metricas': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'falsos_positivos': falsos_positivos,
                'falsos_negativos': falsos_negativos
            },
            'analisis_scores': analisis_scores,
            'estadisticas_dia': estadisticas_dia
        }
        
        # Guardar configuración actualizada (NO el modelo, solo la ml.config)
        with open(ml.MODELS_PATH / "config_modelo_general.pkl", 'wb') as f:
            pickle.dump(ml.config, f)
        
        # Crear backup de configuración
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = ml.MODELS_PATH / f"backup_config_diaria_{timestamp}.pkl"
        with open(backup_path, 'wb') as f:
            pickle.dump(ml.config, f)
        
        # Calcular tiempo de procesamiento
        tiempo_fin = datetime.now()
        tiempo_procesamiento = (tiempo_fin - tiempo_inicio).total_seconds()
        
        # Actualizar registro en Oracle
        if oracle.oracle_pool and id_actualizacion:
            actualizar_fin_actualizacion_diaria(
                id_actualizacion=id_actualizacion,
                umbral_nuevo=umbral_nuevo,
                cambio_porcentual=((umbral_nuevo - estado_inicial['umbral']) / estado_inicial['umbral'] * 100),
                razon_cambio=razon_cambio,
                metricas_performance={
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'falsos_positivos': falsos_positivos,
                    'falsos_negativos': falsos_negativos
                },
                analisis_scores=analisis_scores,
                tiempo_procesamiento=tiempo_procesamiento,
                backup_path=str(backup_path),
                estado="COMPLETADO"
            )
        
        # Limpiar archivo temporal
        os.remove(archivo_path)
        
        logger.info(f"""
        Actualización diaria completada:
        - Registros analizados: {len(df)}
        - Umbral anterior: {estado_inicial['umbral']:.4f}
        - Umbral nuevo: {umbral_nuevo:.4f}
        - Cambio: {((umbral_nuevo - estado_inicial['umbral']) / estado_inicial['umbral'] * 100):.2f}%
        - Precisión: {precision:.2f}
        - Recall: {recall:.2f}
        - F1-Score: {f1_score:.2f}
        - Tiempo: {tiempo_procesamiento:.2f} segundos
        """)
        
    except Exception as e:
        logger.error(f"Error en actualización diaria: {str(e)}")
        
        # Actualizar registro de error en Oracle
        if oracle.oracle_pool and id_actualizacion:
            try:
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE APRENDIZAJE_INCREMENTAL
                        SET ESTADO = 'ERROR',
                            MENSAJE_ERROR = :1,
                            TIEMPO_PROCESAMIENTO_SEG = :2
                        WHERE ID_APRENDIZAJE = :3
                    """, [str(e), (datetime.now() - tiempo_inicio).total_seconds(), id_actualizacion])
                    conn.commit()
            except:
                pass
        
        if os.path.exists(archivo_path):
            os.remove(archivo_path)

# # ## 7. ENDPOINTS DE GESTIÓN DEL MODELO
# 

# In[14]:


@router.get("/modelo/estadisticas", response_model=EstadisticasModelo, tags=["Modelo"])
async def obtener_estadisticas_modelo():
    """
    Obtiene las estadísticas actuales del modelo
    """
    if not ml.modelo or not ml.config:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    return EstadisticasModelo(
        fecha_entrenamiento=ml.config.get('fecha_entrenamiento', 'No disponible'),
        fecha_ultima_actualizacion=ml.config.get('fecha_ultima_actualizacion'),
        umbral_global=ml.umbral_global,
        n_trees=ml.config.get('n_trees', 0),
        tree_height=ml.config.get('tree_height', 0),
        registros_entrenamiento=ml.config.get('registros_entrenamiento', 0),
        paises_entrenamiento=ml.config.get('paises_entrenamiento', 0),
        paises_conocidos=len(ml.stats_dict)
    )

@router.put("/modelo/actualizar-umbral", tags=["Modelo"])
async def actualizar_umbral(configuracion: ConfiguracionUmbral):
    """
    Actualiza el umbral de detección de anomalías
    """
    if not ml.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Guardar umbral anterior
        umbral_anterior = ml.umbral_global

        # Actualizar umbral
        ml.umbral_global = configuracion.nuevo_umbral
        ml.config['umbral_global'] = ml.umbral_global

        # Guardar configuración actualizada
        with open(ml.MODELS_PATH / "config_modelo_general.pkl", 'wb') as f:
            pickle.dump(ml.config, f)

        # NUEVO: Guardar en Oracle si está disponible
        if oracle.oracle_pool:
            try:
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO HISTORICO_UMBRALES 
                        (UMBRAL_ANTERIOR, UMBRAL_NUEVO, RAZON_CAMBIO, USUARIO)
                        VALUES (:1, :2, :3, :4)
                    """, [umbral_anterior, ml.umbral_global, configuracion.razon, "API"])
                    conn.commit()
                    logger.info("Cambio de umbral registrado en Oracle")
            except Exception as e:
                logger.error(f"Error guardando cambio de umbral en Oracle: {e}")

        # Guardar también en archivo JSON (como respaldo)
        log_cambio = {
            'timestamp': datetime.now(),
            'umbral_anterior': umbral_anterior,
            'umbral_nuevo': ml.umbral_global,
            'razon': configuracion.razon
        }
        log_file = str(ml.LOG_PATH / f"cambios_umbral_{datetime.now():%Y%m}.json")

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_cambio)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)

        return {
            "mensaje": "Umbral actualizado exitosamente",
            "umbral_anterior": umbral_anterior,
            "umbral_nuevo": ml.umbral_global,
            "timestamp": datetime.now(),
            "guardado_oracle": oracle.oracle_pool is not None
        }

    except Exception as e:
        logger.error(f"Error actualizando umbral: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/modelo/recargar", tags=["Modelo"])
async def recargar_modelo():
    """
    Recarga el modelo desde disco (útil si se actualizó externamente)
    """
    try:
        if ml.cargar_modelo():
            return {
                "mensaje": "Modelo recargado exitosamente",
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(status_code=500, detail="Error al recargar modelo")
    except Exception as e:
        logger.error(f"Error recargando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/modelo/historial-umbrales", tags=["Modelo"])
async def historial_umbrales(dias: int = Query(30, description="Días de historial")):
    """
    Muestra la evolución del umbral en los últimos días
    """
    if not oracle.oracle_pool:
        # Si no hay Oracle, buscar en archivos de backup
        backups = []
        for archivo in os.listdir(ml.MODELS_PATH):
            if archivo.startswith("backup_config_diaria_"):
                fecha_str = archivo.split("_")[3].split(".")[0]
                try:
                    fecha = datetime.strptime(fecha_str[:8], "%Y%m%d")
                    
                    with open(ml.MODELS_PATH / archivo, 'rb') as f:
                        config_backup = pickle.load(f)
                    
                    if 'ultima_actualizacion_diaria' in config_backup:
                        info = config_backup['ultima_actualizacion_diaria']
                        backups.append({
                            'fecha': fecha.strftime("%Y-%m-%d"),
                            'umbral': config_backup['umbral_global'],
                            'cambio_porcentual': info.get('cambio_porcentual', 0),
                            'precision': info['metricas']['precision'],
                            'recall': info['metricas']['recall'],
                            'f1_score': info['metricas']['f1_score']
                        })
                except:
                    continue
        
        return {
            "historial": sorted(backups, key=lambda x: x['fecha'], reverse=True)[:dias],
            "fuente": "archivos_backup"
        }
    
    # Si hay Oracle, consultar de la BD
    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    FECHA_EJECUCION,
                    UMBRAL_ANTERIOR,
                    UMBRAL_NUEVO,
                    TIEMPO_PROCESAMIENTO_SEG,
                    ESTADO
                FROM APRENDIZAJE_INCREMENTAL
                WHERE FECHA_EJECUCION >= SYSDATE - :dias
                ORDER BY FECHA_EJECUCION DESC
            """, [dias])
            
            historial = []
            for row in cursor:
                historial.append({
                    'fecha': row[0].strftime("%Y-%m-%d %H:%M"),
                    'umbral_anterior': float(row[1]) if row[1] else None,
                    'umbral_nuevo': float(row[2]) if row[2] else None,
                    'cambio': float(row[2] - row[1]) if row[1] and row[2] else None,
                    'tiempo_seg': float(row[3]) if row[3] else None,
                    'estado': row[4]
                })
            
            return {
                "historial": historial,
                "fuente": "oracle",
                "umbral_actual": ml.umbral_global
            }
            
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ## 8. ENDPOINTS DE ANÁLISIS Y REPORTES

# In[15]:


@router.get("/analisis/resumen-diario", tags=["Análisis"])
async def resumen_diario(fecha: str = Query(..., description="Fecha en formato YYYY-MM-DD")):
    """
    Obtiene un resumen de las anomalías detectadas en un día específico
    """
    try:
        # Convertir fecha
        fecha_obj = datetime.strptime(fecha, "%Y-%m-%d")

        # Buscar archivos de resultados del día
        archivos_dia = [
            f for f in os.listdir(ml.OUTPUT_PATH)
            if f.startswith("resultados_") and fecha.replace("-", "") in f
        ]

        if not archivos_dia:
            return {
                "fecha": fecha,
                "registros_procesados": 0,
                "anomalias_detectadas": 0,
                "mensaje": "No hay datos para esta fecha"
            }

        # Cargar y combinar datos
        dfs = [pd.read_csv(str(ml.OUTPUT_PATH / f)) for f in archivos_dia]
        df_total = pd.concat(dfs, ignore_index=True)

        # Calcular estadísticas
        total_registros = len(df_total)
        total_anomalias = int(df_total['es_anomalia'].sum())

        # Distribución por tipo
        distribucion_tipos = {}
        anomalias = df_total[df_total['es_anomalia'] == True]
        for tipo, cnt in anomalias['tipo_anomalia'].value_counts().items():
            distribucion_tipos[tipo] = {
                "cantidad": int(cnt),
                "porcentaje": round(cnt / len(anomalias) * 100, 2)
            }

        # Top países
        top_paises = [
            {"pais": pais, "anomalias": int(cnt)}
            for pais, cnt in anomalias['CODIGODEPAIS'].value_counts().head(10).items()
        ]

        return {
            "fecha": fecha,
            "registros_procesados": total_registros,
            "anomalias_detectadas": total_anomalias,
            "tasa_anomalias": round(total_anomalias / total_registros * 100, 2) if total_registros > 0 else 0,
            "distribucion_tipos": distribucion_tipos,
            "top_paises_anomalias": top_paises
        }

    except Exception as e:
        logger.error(f"Error generando resumen diario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analisis/tendencias", tags=["Análisis"])
async def obtener_tendencias(dias: int = Query(7, description="Número de días a analizar")):
    """
    Obtiene las tendencias de anomalías de los últimos N días
    """
    try:
        fecha_fin = datetime.now()
        fecha_inicio = fecha_fin - timedelta(days=dias)
        tendencias = []

        for i in range(dias):
            fecha_i = fecha_inicio + timedelta(days=i)
            archivos = [
                f for f in os.listdir(ml.OUTPUT_PATH)
                if f.startswith("resultados_") and fecha_i.strftime("%Y%m%d") in f
            ]

            if archivos:
                dfs = [pd.read_csv(str(ml.OUTPUT_PATH / f)) for f in archivos]
                df_dia = pd.concat(dfs, ignore_index=True)
                anom_count = int(df_dia['es_anomalia'].sum())
                tasas = round(df_dia['es_anomalia'].mean() * 100, 2) if len(df_dia) > 0 else 0
                registros = len(df_dia)
            else:
                anom_count = 0
                tasas = 0
                registros = 0

            tendencias.append({
                "fecha": fecha_i.strftime("%Y-%m-%d"),
                "registros": registros,
                "anomalias": anom_count,
                "tasa": tasas
            })

        registros_totales = sum(t['registros'] for t in tendencias)
        anom_totales = sum(t['anomalias'] for t in tendencias)

        return {
            "periodo": f"Últimos {dias} días",
            "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
            "fecha_fin": fecha_fin.strftime("%Y-%m-%d"),
            "tendencias": tendencias,
            "resumen": {
                "registros_totales": registros_totales,
                "anomalias_totales": anom_totales,
                "tasa_promedio": round(anom_totales / registros_totales * 100, 2) if registros_totales > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"Error obteniendo tendencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analisis/exportar-anomalias", tags=["Análisis"])
async def exportar_anomalias(
    fecha_inicio: str = Query(..., description="Fecha inicio YYYY-MM-DD"),
    fecha_fin: str    = Query(..., description="Fecha fin YYYY-MM-DD")
):
    """
    Exporta todas las anomalías detectadas en un rango de fechas
    """
    try:
        # Convertir fechas
        inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        fin    = datetime.strptime(fecha_fin,    "%Y-%m-%d")

        if inicio > fin:
            raise HTTPException(status_code=400, detail="Fecha inicio debe ser anterior a fecha fin")

        anomalias_totales = []
        for archivo in os.listdir(ml.OUTPUT_PATH):
            if archivo.startswith("resultados_"):
                fecha_arch = archivo.split("_")[2][:8]
                try:
                    fecha_obj = datetime.strptime(fecha_arch, "%Y%m%d")
                except ValueError:
                    continue

                if inicio <= fecha_obj <= fin:
                    df = pd.read_csv(str(ml.OUTPUT_PATH / archivo))
                    anomalias_totales.append(df[df['es_anomalia'] == True])

        if not anomalias_totales:
            return {
                "mensaje": "No se encontraron anomalías en el rango especificado",
                "fecha_inicio": fecha_inicio,
                "fecha_fin": fecha_fin
            }

        # Combinar y exportar
        df_out = pd.concat(anomalias_totales, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(ml.OUTPUT_PATH / f"exportacion_anomalias_{timestamp}.csv")
        df_out.to_csv(out_path, index=False)

        return FileResponse(out_path, media_type="text/csv",
                            filename=f"anomalias_{fecha_inicio}_{fecha_fin}.csv")

    except Exception as e:
        logger.error(f"Error exportando anomalías: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ## 9. ENDPOINTS DE MONITOREO Y SALUD

# In[16]:


@router.get("/salud", response_model=EstadoSistema, tags=["Monitoreo"])
async def verificar_salud():
    """
    Verifica el estado de salud del sistema
    """
    try:
        # Calcular estadísticas del día
        hoy = datetime.now().strftime("%Y%m%d")
        registros_hoy = 0
        anomalias_hoy = 0
        for archivo in os.listdir(ml.OUTPUT_PATH):
            if archivo.startswith("resultados_") and hoy in archivo:
                df = pd.read_csv(str(ml.OUTPUT_PATH / archivo))
                registros_hoy += len(df)
                anomalias_hoy += int(df['es_anomalia'].sum())

        # Espacio en disco
        stat = shutil.disk_usage(ml.OUTPUT_PATH)
        espacio_gb = stat.free / (1024**3)

        # Último procesamiento
        archivos = [f for f in os.listdir(ml.OUTPUT_PATH) if f.startswith("resultados_")]
        ultimo_procesamiento = None
        if archivos:
            ultimo_archivo = max(archivos)
            # Extraer timestamp del nombre del archivo
            try:
                parts = ultimo_archivo.split("_")
                timestamp_str = parts[2] + parts[3].replace(".csv", "")
                ultimo_procesamiento = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            except:
                pass

        return EstadoSistema(
            estado="saludable" if ml.modelo else "modelo no cargado",
            modelo_cargado=bool(ml.modelo),
            ultimo_procesamiento=ultimo_procesamiento,
            registros_procesados_hoy=registros_hoy,
            anomalias_detectadas_hoy=anomalias_hoy,
            espacio_disco_gb=round(espacio_gb, 2)
        )

    except Exception as e:
        logger.error(f"Error verificando salud: {e}")
        return EstadoSistema(
            estado="error",
            modelo_cargado=False,
            ultimo_procesamiento=None,
            registros_procesados_hoy=0,
            anomalias_detectadas_hoy=0,
            espacio_disco_gb=0
        )


@router.get("/logs/recientes", tags=["Monitoreo"])
async def obtener_logs_recientes(limite: int = Query(100, description="Número de logs a retornar")):
    """
    Obtiene los logs más recientes del sistema
    """
    try:
        logs = []
        # Leer todos los JSON de logs
        for archivo in os.listdir(ml.LOG_PATH):
            if archivo.endswith(".json"):
                with open(str(ml.LOG_PATH / archivo), "r") as f:
                    contenido = json.load(f)
                if isinstance(contenido, list):
                    logs.extend(contenido)
                else:
                    logs.append(contenido)

        # Ordenar y recortar
        logs_ordenados = sorted(
            logs,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:limite]

        return {
            "total_logs": len(logs_ordenados),
            "logs": logs_ordenados
        }

    except Exception as e:
        logger.error(f"Error obteniendo logs: {e}")
        return {"total_logs": 0, "logs": []}


@router.delete("/limpiar/archivos-antiguos", tags=["Monitoreo"])
async def limpiar_archivos_antiguos(dias_retener: int = Query(7, description="Días a retener")):
    """
    Limpia archivos más antiguos que los días especificados
    """
    try:
        fecha_limite = datetime.now() - timedelta(days=dias_retener)
        archivos_eliminados = 0
        espacio_liberado = 0

        # ml.OUTPUT_PATH
        for archivo in os.listdir(ml.OUTPUT_PATH):
            ruta = str(ml.OUTPUT_PATH / archivo)
            if os.path.isfile(ruta):
                mod_time = datetime.fromtimestamp(os.path.getmtime(ruta))
                if mod_time < fecha_limite:
                    tamaño = os.path.getsize(ruta)
                    os.remove(ruta)
                    archivos_eliminados += 1
                    espacio_liberado += tamaño

        # ml.TEMP_PATH
        for archivo in os.listdir(ml.TEMP_PATH):
            ruta = str(ml.TEMP_PATH / archivo)
            if os.path.isfile(ruta):
                mod_time = datetime.fromtimestamp(os.path.getmtime(ruta))
                if mod_time < fecha_limite:
                    tamaño = os.path.getsize(ruta)
                    os.remove(ruta)
                    archivos_eliminados += 1
                    espacio_liberado += tamaño

        espacio_mb = espacio_liberado / (1024**2)
        return {
            "mensaje": "Limpieza completada",
            "archivos_eliminados": archivos_eliminados,
            "espacio_liberado_mb": round(espacio_mb, 2),
            "fecha_limite": fecha_limite.strftime("%Y-%m-%d")
        }

    except Exception as e:
        logger.error(f"Error limpiando archivos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ## 10. ENDPOINTS DE INFORMACIÓN DE PAÍSES

# In[17]:


@router.get("/paises/lista", tags=["Información"])
async def listar_paises():
    """
    Lista todos los países conocidos por el modelo
    """
    if not ml.stats_dict:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    paises = []
    for pais, stats in ml.stats_dict.items():
        paises.append({
            "codigo": pais,
            "categoria": stats.get("CATEGORIA", "Desconocido"),
            "registros_historicos": stats.get("REGISTROS", 0),
            "promedio_llamadas": round(stats.get("LLAMADAS_MEAN", 0), 2),
            "promedio_minutos": round(stats.get("MINUTOS_MEAN", 0), 2),
            "promedio_destinos": round(stats.get("DESTINOS_MEAN", 0), 2),
        })

    return {
        "total_paises": len(paises),
        "paises": sorted(
            paises,
            key=lambda x: x["registros_historicos"],
            reverse=True
        )
    }


@router.get("/paises/{codigo_pais}/estadisticas", tags=["Información"])
async def estadisticas_pais(codigo_pais: str):
    """
    Obtiene estadísticas detalladas de un país específico
    """
    if not ml.stats_dict:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if codigo_pais not in ml.stats_dict:
        raise HTTPException(
            status_code=404,
            detail=f"País {codigo_pais} no encontrado"
        )

    stats = ml.stats_dict[codigo_pais]

    return {
        "codigo_pais": codigo_pais,
        "categoria": stats.get("CATEGORIA"),
        "contexto_historico": ml.contexto_historico.get(codigo_pais, "No disponible"),
        "estadisticas": {
            "registros": stats.get("REGISTROS", 0),
            "llamadas": {
                "promedio": round(stats.get("LLAMADAS_MEAN", 0), 2),
                "desviacion": round(stats.get("LLAMADAS_STD", 0), 2),
                "percentil_90": round(stats.get("LLAMADAS_P90", 0), 2),
                "percentil_95": round(stats.get("LLAMADAS_P95", 0), 2),
            },
            "minutos": {
                "promedio": round(stats.get("MINUTOS_MEAN", 0), 2),
                "desviacion": round(stats.get("MINUTOS_STD", 0), 2),
                "percentil_90": round(stats.get("MINUTOS_P90", 0), 2),
                "percentil_95": round(stats.get("MINUTOS_P95", 0), 2),
            },
            "destinos": {
                "promedio": round(stats.get("DESTINOS_MEAN", 0), 2),
                "desviacion": round(stats.get("DESTINOS_STD", 0), 2),
                "percentil_90": round(stats.get("DESTINOS_P90", 0), 2),
                "percentil_95": round(stats.get("DESTINOS_P95", 0), 2),
            },
        }
    }


# # ##NUEVO ENDPOINT PARA CONSULTAR APRENDISAJE INCREMENTAL

# In[18]:


@router.get("/actualizacion/historial", tags=["Actualización Diaria"])
async def historial_actualizacion(
    limite: int = Query(30, description="Número de registros a mostrar")
):
    """
    Obtiene el historial de aprendizajes incrementales realizados
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    ID_APRENDIZAJE,
                    FECHA_EJECUCION,
                    FECHA_DATOS,
                    TOTAL_REGISTROS,
                    REGISTROS_PROCESADOS,
                    PAISES_NUEVOS,
                    PAISES_ACTUALIZADOS,
                    UMBRAL_ANTERIOR,
                    UMBRAL_NUEVO,
                    PAISES_CON_CAMBIO_CATEGORIA,
                    TIEMPO_PROCESAMIENTO_SEG,
                    ESTADO
                FROM APRENDIZAJE_INCREMENTAL
                ORDER BY FECHA_EJECUCION DESC
                FETCH FIRST :limite ROWS ONLY
            """, [limite])

            columnas = [col[0] for col in cursor.description]
            historiales = []

            for row in cursor:
                hist = dict(zip(columnas, row))
                # Convertir timestamps
                if hist['FECHA_EJECUCION']:
                    hist['FECHA_EJECUCION'] = hist['FECHA_EJECUCION'].isoformat()
                if hist['FECHA_DATOS']:
                    hist['FECHA_DATOS'] = hist['FECHA_DATOS'].isoformat()
                historiales.append(hist)

            return {
                "total": len(historiales),
                "historiales": historiales
            }

    except Exception as e:
        logger.error(f"Error obteniendo historial de aprendizaje: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/actualizacion/{id_actualizacion}/detalle", tags=["Actualización Diaria"])
async def detalle_aprendizaje(id_aprendizaje: int):
    """
    Obtiene el detalle completo de un aprendizaje incremental específico
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM APRENDIZAJE_INCREMENTAL
                WHERE ID_APRENDIZAJE = :id
            """, [id_aprendizaje])

            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Aprendizaje no encontrado")

            columnas = [col[0] for col in cursor.description]
            detalle = dict(zip(columnas, row))

            # Convertir timestamps y CLOB
            if detalle['FECHA_EJECUCION']:
                detalle['FECHA_EJECUCION'] = detalle['FECHA_EJECUCION'].isoformat()
            if detalle['FECHA_DATOS']:
                detalle['FECHA_DATOS'] = detalle['FECHA_DATOS'].isoformat()

            # Parsear JSON de cambios de categoría si existe
            if detalle['DETALLE_CAMBIOS_CATEGORIA']:
                try:
                    detalle['DETALLE_CAMBIOS_CATEGORIA'] = json.loads(
                        detalle['DETALLE_CAMBIOS_CATEGORIA'].read()
                    )
                except:
                    detalle['DETALLE_CAMBIOS_CATEGORIA'] = {}

            return detalle

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo detalle de aprendizaje: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/actualizacion/estadisticas-evolucion", tags=["Actualización Diaria"])
async def estadisticas_evolucion():
    """
    Muestra la evolución del modelo a través de los aprendizajes incrementales
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            # Evolución del umbral
            cursor.execute("""
                SELECT 
                    TO_CHAR(FECHA_EJECUCION, 'YYYY-MM-DD') as fecha,
                    UMBRAL_ANTERIOR,
                    UMBRAL_NUEVO
                FROM APRENDIZAJE_INCREMENTAL
                WHERE ESTADO = 'COMPLETADO'
                ORDER BY FECHA_EJECUCION
            """)

            evolucion_umbral = [
                {
                    "fecha": row[0],
                    "umbral_anterior": float(row[1]) if row[1] else None,
                    "umbral_nuevo": float(row[2]) if row[2] else None
                }
                for row in cursor.fetchall()
            ]

            # Evolución de países conocidos
            cursor.execute("""
                SELECT 
                    TO_CHAR(FECHA_EJECUCION, 'YYYY-MM-DD') as fecha,
                    PAISES_CONOCIDOS_ANTERIOR,
                    PAISES_CONOCIDOS_NUEVO,
                    PAISES_NUEVOS
                FROM APRENDIZAJE_INCREMENTAL
                WHERE ESTADO = 'COMPLETADO'
                ORDER BY FECHA_EJECUCION
            """)

            evolucion_paises = [
                {
                    "fecha": row[0],
                    "paises_antes": row[1],
                    "paises_despues": row[2],
                    "paises_nuevos": row[3]
                }
                for row in cursor.fetchall()
            ]

            # Estadísticas agregadas
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_ejecuciones,
                    SUM(REGISTROS_PROCESADOS) as total_registros,
                    SUM(PAISES_NUEVOS) as total_paises_nuevos,
                    AVG(TIEMPO_PROCESAMIENTO_SEG) as tiempo_promedio,
                    MAX(FECHA_EJECUCION) as ultima_ejecucion
                FROM APRENDIZAJE_INCREMENTAL
                WHERE ESTADO = 'COMPLETADO'
            """)

            stats = cursor.fetchone()
            estadisticas_generales = {
                "total_ejecuciones": stats[0] or 0,
                "total_registros_procesados": stats[1] or 0,
                "total_paises_descubiertos": stats[2] or 0,
                "tiempo_promedio_segundos": float(stats[3]) if stats[3] else 0,
                "ultima_actualizacion": stats[4].isoformat() if stats[4] else None
            }

            return {
                "estadisticas_generales": estadisticas_generales,
                "evolucion_umbral": evolucion_umbral,
                "evolucion_paises": evolucion_paises
            }

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de evolución: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actualizacion/programar-diario", tags=["Actualización Diaria"])
async def programar_aprendizaje_diario(
    hora_ejecucion: str = Query("02:00", description="Hora de ejecución diaria (HH:MM)"),
    directorio_datos: str = Query(..., description="Directorio donde se encuentran los datos diarios")
):
    """
    Programa el aprendizaje incremental diario (requiere implementación de scheduler)
    """
    # Esta es una función placeholder que muestra cómo se podría implementar
    return {
        "mensaje": "Funcionalidad de programación diaria",
        "nota": "Requiere implementar un scheduler (como APScheduler) para ejecución automática",
        "configuracion_sugerida": {
            "hora_ejecucion": hora_ejecucion,
            "directorio_datos": directorio_datos,
            "comando_cron": f"0 {hora_ejecucion.split(':')[1]} {hora_ejecucion.split(':')[0]} * * python ejecutar_aprendizaje_diario.py"
        }
    }


# ## 11. CONFIGURACIÓN PARA EJECUTAR LA API

