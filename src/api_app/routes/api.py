from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import os
import json
import asyncio
import pickle
import shutil
import logging
import oracledb  # Agregar esta importación al inicio del archivo

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
executor = ThreadPoolExecutor(max_workers=2)


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
async def scoring_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Realiza scoring de un archivo CSV de forma asíncrona
    """
    if not ml.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Guardar archivo temporal
        temp_file = str(ml.TEMP_PATH / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validar archivo básico
        df = pd.read_csv(temp_file, nrows=5)  # Solo leer primeras 5 filas para validación
        
        columnas_requeridas = ['FECHA', 'CODIGODEPAIS', 'LINEA', 'N_LLAMADAS', 'N_MINUTOS', 'N_DESTINOS']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

        if columnas_faltantes:
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {columnas_faltantes}")

        # Registrar inicio del lote
        lote_id = None
        if oracle.oracle_pool:
            try:
                lote_id = registrar_lote_procesamiento(
                    archivo_entrada=file.filename,
                    total_registros=0,  # Se actualizará después
                    total_anomalias=0,  # Se actualizará después
                    tasa_anomalias=0,   # Se actualizará después
                    archivo_salida=None
                )
            except Exception as e:
                logger.error(f"Error registrando lote: {e}")

        # Procesar en background de forma asíncrona
        background_tasks.add_task(
            procesar_lote_async, 
            temp_file, 
            file.filename, 
            lote_id
        )

        return {
            "mensaje": "Procesamiento iniciado en segundo plano",
            "lote_id": lote_id,
            "archivo": file.filename,
            "timestamp": datetime.now(),
            "estado": "EN_PROCESO"
        }

    except Exception as e:
        logger.error(f"Error iniciando procesamiento de lote: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

async def procesar_lote_async(archivo_path: str, nombre_archivo: str, lote_id: int):
    """
    Procesa el lote de forma asíncrona sin bloquear el hilo principal
    """
    loop = asyncio.get_event_loop()
    
    try:
        # Ejecutar en thread pool para no bloquear
        await loop.run_in_executor(
            executor, 
            procesar_lote_sincrono, 
            archivo_path, 
            nombre_archivo, 
            lote_id
        )
    except Exception as e:
        logger.error(f"Error en procesamiento asíncrono: {e}")

def procesar_lote_sincrono(archivo_path: str, nombre_archivo: str, lote_id: int):
    """
    Función síncrona que hace el trabajo pesado
    """
    try:
        # Leer CSV completo
        df = pd.read_csv(archivo_path)
        
        if 'CODIGODEPAIS' in df.columns:
            df['CODIGODEPAIS'] = pd.to_numeric(df['CODIGODEPAIS'], errors='coerce').astype('Int64')

        # Procesar registros
        resultados = []
        anomalias_detectadas = 0

        for idx, row in df.iterrows():
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

        # Guardar en Oracle
        if oracle.oracle_pool and lote_id:
            try:
                # Actualizar registro del lote
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE LOTES_PROCESAMIENTO 
                        SET TOTAL_REGISTROS = :1,
                            TOTAL_ANOMALIAS = :2,
                            TASA_ANOMALIAS = :3,
                            TIMESTAMP_FIN = :4,
                            ESTADO = 'COMPLETADO'
                        WHERE ID_LOTE = :5
                    """, [
                        len(resultados),
                        anomalias_detectadas,
                        round(anomalias_detectadas/len(resultados)*100, 2) if len(resultados) > 0 else 0,
                        datetime.now(),
                        lote_id
                    ])
                    conn.commit()

                # Guardar anomalías
                df_anomalias = pd.DataFrame([r for r in resultados if r['es_anomalia']])
                if not df_anomalias.empty:
                    guardar_anomalias_oracle(df_anomalias, nombre_archivo, str(lote_id))

                logger.info(f"Procesamiento completado - Lote ID: {lote_id}, Anomalías: {anomalias_detectadas}")

            except Exception as e:
                logger.error(f"Error guardando en Oracle: {e}")
                # Marcar como error
                if oracle.oracle_pool:
                    with get_oracle_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE LOTES_PROCESAMIENTO 
                            SET ESTADO = 'ERROR', MENSAJE_ERROR = :1
                            WHERE ID_LOTE = :2
                        """, [str(e), lote_id])
                        conn.commit()

        # Limpiar archivo temporal
        if os.path.exists(archivo_path):
            os.remove(archivo_path)
            
    except Exception as e:
        logger.error(f"Error en procesamiento síncrono: {e}")
        
        # Marcar lote como error
        if oracle.oracle_pool and lote_id:
            try:
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE LOTES_PROCESAMIENTO 
                        SET ESTADO = 'ERROR', MENSAJE_ERROR = :1
                        WHERE ID_LOTE = :2
                    """, [str(e), lote_id])
                    conn.commit()
            except:
                pass
        
        if os.path.exists(archivo_path):
            os.remove(archivo_path)


@router.get("/alertas/{id_alerta}/detalle", tags=["Alertas Oracle"])
async def obtener_detalle_alerta(id_alerta: int):
    """
    Obtiene el detalle completo de una alerta específica
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    ID_ALERTA, FECHA_PROCESAMIENTO, FECHA_REGISTRO, CODIGO_PAIS, 
                    LINEA, N_LLAMADAS, N_MINUTOS, N_DESTINOS, SCORE_ANOMALIA,
                    UMBRAL, TIPO_ANOMALIA, TIPO_CONTEXTO, RAZON_DECISION,
                    ARCHIVO_ORIGEN, LOTE_PROCESAMIENTO
                FROM ALERTAS_FRAUDE
                WHERE ID_ALERTA = :id_alerta
            """, [id_alerta])

            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Alerta no encontrada")

            # Obtener nombres de columnas
            columnas = [col[0] for col in cursor.description]
            alerta = dict(zip(columnas, row))
            
            # Convertir datetime a string
            if alerta['FECHA_PROCESAMIENTO']:
                alerta['FECHA_PROCESAMIENTO'] = alerta['FECHA_PROCESAMIENTO'].isoformat()

            return {
                "alerta": alerta
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo detalle de alerta: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/lotes/{id_lote}/descargar", tags=["Alertas Oracle"])
async def descargar_archivo_lote(id_lote: int):
    """
    Genera y descarga las anomalías de un lote específico
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            # Obtener información del lote
            cursor.execute("""
                SELECT ARCHIVO_ENTRADA, TIMESTAMP_INICIO 
                FROM LOTES_PROCESAMIENTO 
                WHERE ID_LOTE = :id_lote
            """, [id_lote])
            
            lote_info = cursor.fetchone()
            if not lote_info:
                raise HTTPException(status_code=404, detail="Lote no encontrado")

            # Obtener alertas del lote
            cursor.execute("""
                SELECT 
                    FECHA_REGISTRO, CODIGO_PAIS, LINEA, N_LLAMADAS, 
                    N_MINUTOS, N_DESTINOS, SCORE_ANOMALIA, UMBRAL,
                    TIPO_ANOMALIA, TIPO_CONTEXTO, RAZON_DECISION
                FROM ALERTAS_FRAUDE
                WHERE LOTE_PROCESAMIENTO = :lote_id
                ORDER BY FECHA_PROCESAMIENTO
            """, [str(id_lote)])

            # Convertir a DataFrame
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)

            if df.empty:
                raise HTTPException(status_code=404, detail="No hay anomalías en este lote")

            # Generar archivo temporal
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_file = ml.TEMP_PATH / f"lote_{id_lote}_{timestamp}.csv"
            df.to_csv(temp_file, index=False)

            return FileResponse(
                path=temp_file,
                media_type="text/csv",
                filename=f"anomalias_lote_{id_lote}.csv"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando archivo de lote: {e}")
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

        features = ml.crear_features_contextualizadas_mejorada(row_dict, ml.stats_dict)
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

        # CAMBIO: usar oracledb.NUMBER en lugar de oracle.NUMBER
        id_var = cursor.var(oracledb.NUMBER)

        cursor.execute("""
            INSERT INTO ACTUALIZACION_DIARIA_UMBRAL (
                FECHA_DATOS, ARCHIVO_ENTRADA, TOTAL_REGISTROS,
                PAISES_TOTALES, UMBRAL_ANTERIOR, N_TREES_ANTERIOR,
                PAISES_CONOCIDOS_ANTERIOR, ESTADO
            ) VALUES (
                TRUNC(SYSDATE), :1, :2, :3, :4, :5, :6, 'EN_PROCESO'
            ) RETURNING ID_ACTUALIZACION INTO :id
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
            f"Cambio: {kwargs.get('razon_cambio', 'N/A')}",
            id_actualizacion
        ])
        
        conn.commit()

# # ## 6. ENDPOINTS DE APRENDIZAJE INCREMENTAL
# 

# In[13]:
def procesar_actualizacion_diaria_sincrono(archivo_path: str, df: pd.DataFrame):
    """
    Versión síncrona que retorna el resultado en lugar de ejecutarse en background
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
        logger.info(f"Iniciando actualización diaria con {len(df)} registros")
        
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
            if pd.isna(row.get('CODIGODEPAIS')):
                continue
                
            row_dict = {
                'CODIGODEPAIS': int(row['CODIGODEPAIS']),
                'N_LLAMADAS': row['N_LLAMADAS'],
                'N_MINUTOS': row['N_MINUTOS'],
                'N_DESTINOS': row['N_DESTINOS']
            }
            
            # Crear features
            features = ml.crear_features_contextualizadas_mejorada(row_dict, ml.stats_dict)
            features_scaled = ml.scaler.transform_one(features)
            
            # Obtener score del modelo River
            score = ml.modelo.score_one(features_scaled)
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
        
        if not scores_dia:
            raise ValueError("No se pudieron procesar registros válidos")
        
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
        umbral_estadistico = analisis_scores['p95']
        
        if len(scores_por_tipo['normales']) > 0:
            umbral_separacion = np.percentile(scores_por_tipo['normales'], 99)
        else:
            umbral_separacion = umbral_estadistico
        
        # 4. DECISIÓN FINAL DEL UMBRAL
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
        
        # 5. ACTUALIZAR CONFIGURACIÓN
        ml.umbral_global = umbral_nuevo
        ml.config['umbral_global'] = ml.umbral_global
        ml.config['fecha_ultima_actualizacion'] = datetime.now().isoformat()
        
        # Guardar configuración actualizada
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
                registros_analizados=len(df),
                umbral_nuevo=umbral_nuevo,
                tiempo_procesamiento=tiempo_procesamiento,
                estado="COMPLETADO",
                backup_path=str(backup_path),
                razon_cambio=razon_cambio
            )
        
        # Limpiar archivo temporal
        if os.path.exists(archivo_path):
            os.remove(archivo_path)
        
        logger.info(f"""
        Actualización diaria completada:
        - Registros analizados: {len(df)}
        - Umbral anterior: {estado_inicial['umbral']:.4f}
        - Umbral nuevo: {umbral_nuevo:.4f}
        - Cambio: {((umbral_nuevo - estado_inicial['umbral']) / estado_inicial['umbral'] * 100):.2f}%
        - Tiempo: {tiempo_procesamiento:.2f} segundos
        """)
        
        # RETORNAR EL RESULTADO
        return {
            'registros_analizados': len(df),
            'umbral_anterior': estado_inicial['umbral'],
            'umbral_nuevo': umbral_nuevo,
            'cambio_porcentual': ((umbral_nuevo - estado_inicial['umbral']) / estado_inicial['umbral'] * 100),
            'razon_cambio': razon_cambio,
            'tiempo_procesamiento': tiempo_procesamiento
        }
        
    except Exception as e:
        logger.error(f"Error en actualización diaria: {str(e)}")
        
        # Actualizar registro de error en Oracle
        if oracle.oracle_pool and id_actualizacion:
            try:
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE ACTUALIZACION_DIARIA_UMBRAL
                        SET ESTADO = 'ERROR',
                            MENSAJE_ERROR = :1,
                            TIEMPO_PROCESAMIENTO_SEG = :2
                        WHERE ID_ACTUALIZACION = :3
                    """, [str(e), (datetime.now() - tiempo_inicio).total_seconds(), id_actualizacion])
                    conn.commit()
            except:
                pass
        
        if os.path.exists(archivo_path):
            os.remove(archivo_path)
        
        raise e

@router.post("/modelo/actualizacion-diaria", tags=["Modelo"])
async def actualizacion_diaria(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Realiza la actualización diaria del umbral de forma asíncrona
    """
    if not ml.modelo:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Guardar archivo temporal
        temp_file = str(ml.TEMP_PATH / f"diario_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validar archivo básico
        df = pd.read_csv(temp_file, nrows=5)
        
        columnas_requeridas = ['CODIGODEPAIS', 'N_LLAMADAS', 'N_MINUTOS', 'N_DESTINOS']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {columnas_faltantes}")

        # Procesar en background
        background_tasks.add_task(
            procesar_actualizacion_async, 
            temp_file
        )
        
        return {
            "mensaje": "Actualización diaria iniciada en segundo plano",
            "archivo": file.filename,
            "timestamp": datetime.now(),
            "estado": "EN_PROCESO"
        }
        
    except Exception as e:
        logger.error(f"Error iniciando actualización diaria: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

async def procesar_actualizacion_async(archivo_path: str):
    """
    Procesa la actualización de forma asíncrona
    """
    loop = asyncio.get_event_loop()
    
    try:
        # Leer archivo completo
        df = pd.read_csv(archivo_path)
        
        # Ejecutar en thread pool
        await loop.run_in_executor(
            executor, 
            procesar_actualizacion_diaria_sincrono_v2, 
            archivo_path, 
            df
        )
    except Exception as e:
        logger.error(f"Error en actualización asíncrona: {e}")

def procesar_actualizacion_diaria_sincrono_v2(archivo_path: str, df: pd.DataFrame):
    """
    Versión síncrona V2 para procesamiento en thread pool sin bloquear FastAPI
    """
    
    tiempo_inicio = datetime.now()
    id_actualizacion = None
    estado_inicial = {
        'umbral': ml.umbral_global,
        'n_trees': ml.config.get('n_trees', 0),
        'paises_conocidos': len(ml.stats_dict)
    }
    
    try:
        logger.info(f"Iniciando actualización diaria asíncrona con {len(df)} registros")
        
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
        
        # Procesar en lotes para evitar memoria excesiva
        batch_size = 1000
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            logger.info(f"Procesando lote {start_idx}-{end_idx} de {total_rows}")
            
            for idx, row in batch_df.iterrows():
                if pd.isna(row.get('CODIGODEPAIS')):
                    continue
                    
                try:
                    row_dict = {
                        'CODIGODEPAIS': int(row['CODIGODEPAIS']),
                        'N_LLAMADAS': row['N_LLAMADAS'],
                        'N_MINUTOS': row['N_MINUTOS'],
                        'N_DESTINOS': row['N_DESTINOS']
                    }
                    
                    # Crear features
                    features = ml.crear_features_contextualizadas_mejorada(row_dict, ml.stats_dict)
                    features_scaled = ml.scaler.transform_one(features)
                    
                    # Obtener score del modelo River
                    score = ml.modelo.score_one(features_scaled)
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
                        
                except Exception as e:
                    logger.warning(f"Error procesando fila {idx}: {e}")
                    continue
        
        if not scores_dia:
            raise ValueError("No se pudieron procesar registros válidos")
        
        logger.info(f"Procesados {len(scores_dia)} registros con scores válidos")
        
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
        umbral_estadistico = analisis_scores['p95']
        
        if len(scores_por_tipo['normales']) > 0:
            umbral_separacion = np.percentile(scores_por_tipo['normales'], 99)
        else:
            umbral_separacion = umbral_estadistico
        
        # 4. DECISIÓN FINAL DEL UMBRAL
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
        
        # 5. ACTUALIZAR ESTADÍSTICAS POR PAÍS (sin cambiar el modelo)
        estadisticas_dia = {}
        paises_unicos = df['CODIGODEPAIS'].dropna().unique()
        
        for pais in paises_unicos:
            try:
                pais = int(pais)
                datos_pais = df[df['CODIGODEPAIS'] == pais]
                
                estadisticas_dia[pais] = {
                    'registros': len(datos_pais),
                    'llamadas_mean': float(datos_pais['N_LLAMADAS'].mean()),
                    'minutos_mean': float(datos_pais['N_MINUTOS'].mean()),
                    'destinos_mean': float(datos_pais['N_DESTINOS'].mean()),
                    'llamadas_max': int(datos_pais['N_LLAMADAS'].max()),
                    'minutos_max': float(datos_pais['N_MINUTOS'].max()),
                    'destinos_max': int(datos_pais['N_DESTINOS'].max())
                }
            except Exception as e:
                logger.warning(f"Error procesando estadísticas del país {pais}: {e}")
                continue
        
        # 6. ACTUALIZAR CONFIGURACIÓN
        umbral_anterior = ml.umbral_global
        ml.umbral_global = umbral_nuevo
        ml.config['umbral_global'] = ml.umbral_global
        ml.config['fecha_ultima_actualizacion'] = datetime.now().isoformat()
        ml.config['ultima_actualizacion_diaria'] = {
            'fecha': datetime.now().isoformat(),
            'registros_analizados': len(df),
            'umbral_anterior': umbral_anterior,
            'umbral_nuevo': umbral_nuevo,
            'cambio_porcentual': ((umbral_nuevo - umbral_anterior) / umbral_anterior * 100),
            'razon_cambio': razon_cambio,
            'analisis_scores': analisis_scores,
            'estadisticas_dia': estadisticas_dia
        }
        
        # Guardar configuración actualizada (NO el modelo, solo la config)
        try:
            with open(ml.MODELS_PATH / "config_modelo_general.pkl", 'wb') as f:
                pickle.dump(ml.config, f)
            
            # Crear backup de configuración
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = ml.MODELS_PATH / f"backup_config_diaria_{timestamp}.pkl"
            with open(backup_path, 'wb') as f:
                pickle.dump(ml.config, f)
                
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
        
        # Calcular tiempo de procesamiento
        tiempo_fin = datetime.now()
        tiempo_procesamiento = (tiempo_fin - tiempo_inicio).total_seconds()
        
        # Actualizar registro en Oracle
        if oracle.oracle_pool and id_actualizacion:
            try:
                actualizar_fin_actualizacion_diaria(
                    id_actualizacion=id_actualizacion,
                    registros_analizados=len(df),
                    umbral_nuevo=umbral_nuevo,
                    tiempo_procesamiento=tiempo_procesamiento,
                    estado="COMPLETADO",
                    backup_path=str(backup_path) if 'backup_path' in locals() else '',
                    razon_cambio=razon_cambio
                )
            except Exception as e:
                logger.error(f"Error actualizando Oracle: {e}")
        
        # Limpiar archivo temporal
        if os.path.exists(archivo_path):
            os.remove(archivo_path)
        
        logger.info(f"""
        Actualización diaria completada asíncronamente:
        - Registros analizados: {len(df)}
        - Scores calculados: {len(scores_dia)}
        - Umbral anterior: {umbral_anterior:.4f}
        - Umbral nuevo: {umbral_nuevo:.4f}
        - Cambio: {((umbral_nuevo - umbral_anterior) / umbral_anterior * 100):.2f}%
        - Tiempo total: {tiempo_procesamiento:.2f} segundos
        """)
        
        return {
            'registros_analizados': len(df),
            'scores_calculados': len(scores_dia),
            'umbral_anterior': umbral_anterior,
            'umbral_nuevo': umbral_nuevo,
            'cambio_porcentual': ((umbral_nuevo - umbral_anterior) / umbral_anterior * 100),
            'razon_cambio': razon_cambio,
            'tiempo_procesamiento': tiempo_procesamiento,
            'paises_procesados': len(estadisticas_dia)
        }
        
    except Exception as e:
        logger.error(f"Error en actualización diaria asíncrona: {str(e)}")
        
        # Actualizar registro de error en Oracle
        if oracle.oracle_pool and id_actualizacion:
            try:
                with get_oracle_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE ACTUALIZACION_DIARIA_UMBRAL
                        SET ESTADO = 'ERROR',
                            MENSAJE_ERROR = :1,
                            TIEMPO_PROCESAMIENTO_SEG = :2
                        WHERE ID_ACTUALIZACION = :3
                    """, [str(e), (datetime.now() - tiempo_inicio).total_seconds(), id_actualizacion])
                    conn.commit()
            except Exception as oracle_error:
                logger.error(f"Error actualizando Oracle con error: {oracle_error}")
        
        if os.path.exists(archivo_path):
            os.remove(archivo_path)
        
        raise e

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
            features = ml.crear_features_contextualizadas_mejorada(row_dict, ml.stats_dict)
            features_scaled = ml.scaler.transform_one(features)
            
            # Obtener score del modelo River
            score = ml.modelo.score_one(features_scaled)
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
        
        # 5. ACTUALIZAR ESTADÍSTICAS POR PAÍS (sin cambiar el modelo)
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
        
        # 6. ACTUALIZAR CONFIGURACIÓN
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
                registros_analizados=len(df),
                umbral_nuevo=umbral_nuevo,
                tiempo_procesamiento=tiempo_procesamiento,
                estado="COMPLETADO",
                backup_path=str(backup_path),
                razon_cambio=razon_cambio
            )
        
        # Limpiar archivo temporal
        os.remove(archivo_path)
        
        logger.info(f"""
        Actualización diaria completada:
        - Registros analizados: {len(df)}
        - Umbral anterior: {estado_inicial['umbral']:.4f}
        - Umbral nuevo: {umbral_nuevo:.4f}
        - Cambio: {((umbral_nuevo - estado_inicial['umbral']) / estado_inicial['umbral'] * 100):.2f}%
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
                        UPDATE ACTUALIZACION_DIARIA_UMBRAL
                        SET ESTADO = 'ERROR',
                            MENSAJE_ERROR = :1,
                            TIEMPO_PROCESAMIENTO_SEG = :2
                        WHERE ID_ACTUALIZACION = :3
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
    Muestra la evolución del umbral combinando cambios manuales y automáticos
    """
    if not oracle.oracle_pool:
        return {
            "historial": [],
            "fuente": "oracle_no_disponible",
            "umbral_actual": ml.umbral_global
        }
    
    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            historial_combinado = []
            
            # 1. Obtener cambios de actualizaciones diarias (automáticos)
            cursor.execute("""
                SELECT 
                    FECHA_EJECUCION as FECHA,
                    UMBRAL_ANTERIOR,
                    UMBRAL_NUEVO,
                    TIEMPO_PROCESAMIENTO_SEG,
                    ESTADO,
                    OBSERVACIONES,
                    'AUTOMATICO' as TIPO_CAMBIO,
                    'Sistema' as USUARIO
                FROM ACTUALIZACION_DIARIA_UMBRAL
                WHERE FECHA_EJECUCION >= SYSDATE - :dias
                AND UMBRAL_ANTERIOR IS NOT NULL 
                AND UMBRAL_NUEVO IS NOT NULL
            """, [dias])
            
            for row in cursor:
                historial_combinado.append({
                    'fecha': row[0].strftime('%Y-%m-%d %H:%M') if row[0] else None,
                    'umbral_anterior': float(row[1]) if row[1] else None,
                    'umbral_nuevo': float(row[2]) if row[2] else None,
                    'cambio': float(row[2] - row[1]) if row[1] and row[2] else None,
                    'tiempo_seg': float(row[3]) if row[3] else None,
                    'estado': row[4] or 'COMPLETADO',
                    'razon': row[5] or 'Actualización automática diaria',
                    'tipo_cambio': row[6],  # 'AUTOMATICO'
                    'usuario': row[7],      # 'Sistema'
                    'timestamp_orden': row[0]  # Para ordenar después
                })
            
            # 2. Obtener cambios manuales
            cursor.execute("""
                SELECT 
                    FECHA_CAMBIO as FECHA,
                    UMBRAL_ANTERIOR,
                    UMBRAL_NUEVO,
                    RAZON_CAMBIO,
                    USUARIO,
                    'MANUAL' as TIPO_CAMBIO
                FROM HISTORICO_UMBRALES
                WHERE FECHA_CAMBIO >= SYSDATE - :dias
            """, [dias])
            
            for row in cursor:
                historial_combinado.append({
                    'fecha': row[0].strftime('%Y-%m-%d %H:%M') if row[0] else None,
                    'umbral_anterior': float(row[1]) if row[1] else None,
                    'umbral_nuevo': float(row[2]) if row[2] else None,
                    'cambio': float(row[2] - row[1]) if row[1] and row[2] else None,
                    'tiempo_seg': None,  # Los cambios manuales no tienen tiempo de procesamiento
                    'estado': 'COMPLETADO',
                    'razon': row[3] or 'Cambio manual',
                    'tipo_cambio': row[5],  # 'MANUAL'
                    'usuario': row[4] or 'Usuario',
                    'timestamp_orden': row[0]  # Para ordenar después
                })
            
            # 3. Ordenar por fecha descendente (más recientes primero)
            historial_combinado.sort(
                key=lambda x: x['timestamp_orden'] if x['timestamp_orden'] else datetime.min, 
                reverse=True
            )
            
            # 4. Remover el campo temporal de ordenamiento
            for item in historial_combinado:
                del item['timestamp_orden']
            
            return {
                "historial": historial_combinado,
                "fuente": "oracle_combinado",
                "umbral_actual": ml.umbral_global,
                "total_cambios": len(historial_combinado)
            }
            
    except Exception as e:
        logger.error(f"Error obteniendo historial combinado: {e}")
        return {
            "historial": [],
            "fuente": "error",
            "umbral_actual": ml.umbral_global,
            "error": str(e)
        }

@router.get("/lotes/{lote_id}/estado", tags=["Scoring"])
async def obtener_estado_lote(lote_id: int):
    """
    Consulta el estado actual de un lote en procesamiento
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")
    
    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ID_LOTE, ESTADO, TOTAL_REGISTROS, TOTAL_ANOMALIAS, 
                    TASA_ANOMALIAS, TIMESTAMP_INICIO, TIMESTAMP_FIN, MENSAJE_ERROR
                FROM LOTES_PROCESAMIENTO
                WHERE ID_LOTE = :lote_id
            """, [lote_id])
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Lote no encontrado")
            
            return {
                "lote_id": row[0],
                "estado": row[1],
                "total_registros": row[2] or 0,
                "total_anomalias": row[3] or 0,
                "tasa_anomalias": row[4] or 0,
                "timestamp_inicio": row[5].isoformat() if row[5] else None,
                "timestamp_fin": row[6].isoformat() if row[6] else None,
                "mensaje_error": row[7],
                "completado": row[1] in ['COMPLETADO', 'ERROR']
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error consultando estado de lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/modelo/actualizacion-diaria/estado", tags=["Modelo"])
async def obtener_estado_actualizacion():
    """
    Consulta el estado de la última actualización diaria
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")
    
    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ID_ACTUALIZACION, FECHA_EJECUCION, ESTADO, 
                    TOTAL_REGISTROS, REGISTROS_PROCESADOS,
                    UMBRAL_ANTERIOR, UMBRAL_NUEVO, 
                    TIEMPO_PROCESAMIENTO_SEG, MENSAJE_ERROR
                FROM ACTUALIZACION_DIARIA_UMBRAL
                WHERE FECHA_EJECUCION >= SYSDATE - 1
                ORDER BY FECHA_EJECUCION DESC
                FETCH FIRST 1 ROW ONLY
            """)
            
            row = cursor.fetchone()
            if not row:
                return {
                    "estado": "SIN_ACTUALIZACIONES",
                    "mensaje": "No hay actualizaciones recientes"
                }
            
            return {
                "id_actualizacion": row[0],
                "fecha_ejecucion": row[1].isoformat() if row[1] else None,
                "estado": row[2],
                "total_registros": row[3] or 0,
                "registros_procesados": row[4] or 0,
                "umbral_anterior": float(row[5]) if row[5] else None,
                "umbral_nuevo": float(row[6]) if row[6] else None,
                "tiempo_procesamiento": float(row[7]) if row[7] else None,
                "mensaje_error": row[8],
                "completado": row[2] in ['COMPLETADO', 'ERROR'],
                "progreso": (row[4] / max(row[3], 1)) * 100 if row[3] and row[4] else 0
            }
            
    except Exception as e:
        logger.error(f"Error consultando estado de actualización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
async def obtener_tendencias_fraude(
    dias: int = Query(default=7, description="Número de días a analizar")
):
    """Obtiene tendencias de fraude para los últimos N días desde BD"""
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")
        
    try:
        with oracle.get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT 
                TRUNC(FECHA_PROCESAMIENTO) as FECHA,
                COUNT(*) as ANOMALIAS
            FROM ALERTAS_FRAUDE
            WHERE FECHA_PROCESAMIENTO >= SYSDATE - :dias
            GROUP BY TRUNC(FECHA_PROCESAMIENTO)
            ORDER BY FECHA DESC
            """
            
            cursor.execute(query, [dias])
            
            tendencias = []
            for row in cursor:
                # Obtener total de registros procesados ese día
                cursor_total = conn.cursor()
                cursor_total.execute("""
                    SELECT SUM(TOTAL_REGISTROS) as TOTAL
                    FROM LOTES_PROCESAMIENTO
                    WHERE TRUNC(TIMESTAMP_INICIO) = :fecha
                """, [row[0]])
                total_row = cursor_total.fetchone()
                total = total_row[0] if total_row and total_row[0] else 0
                
                tendencias.append({
                    "fecha": row[0].strftime('%Y-%m-%d'),
                    "anomalias": row[1],
                    "total": total,
                    "tasa": round((row[1] / total * 100) if total > 0 else 0, 2)
                })
                
        return {"tendencias": tendencias}
        
    except Exception as e:
        logger.error(f"Error obteniendo tendencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analisis/exportar-anomalias", tags=["Análisis"])
async def exportar_anomalias(
    formato: str = Query("csv", description="Formato de exportación: csv o excel"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha inicio YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha fin YYYY-MM-DD"),
    limite: int = Query(10000, description="Límite de registros")
):
    """Exporta las anomalías detectadas en formato CSV o Excel desde BD"""
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")
        
    try:
        with oracle.get_oracle_connection() as conn:
            # Construir query con filtros
            where_clause = "WHERE 1=1"
            params = {"limite": limite}
            
            if fecha_inicio:
                where_clause += " AND FECHA_PROCESAMIENTO >= TO_DATE(:fecha_inicio, 'YYYY-MM-DD')"
                params['fecha_inicio'] = fecha_inicio
            if fecha_fin:
                where_clause += " AND FECHA_PROCESAMIENTO <= TO_DATE(:fecha_fin, 'YYYY-MM-DD') + 1"
                params['fecha_fin'] = fecha_fin
                
            query = f"""
            SELECT 
                ID_ALERTA,
                TO_CHAR(FECHA_PROCESAMIENTO, 'YYYY-MM-DD HH24:MI:SS') as FECHA_PROCESAMIENTO,
                FECHA_REGISTRO,
                CODIGO_PAIS,
                LINEA,
                N_LLAMADAS,
                N_MINUTOS,
                N_DESTINOS,
                SCORE_ANOMALIA,
                UMBRAL,
                TIPO_ANOMALIA,
                TIPO_CONTEXTO,
                RAZON_DECISION,
                ARCHIVO_ORIGEN,
                LOTE_PROCESAMIENTO
            FROM ALERTAS_FRAUDE
            {where_clause}
            ORDER BY FECHA_PROCESAMIENTO DESC
            FETCH FIRST :limite ROWS ONLY
            """
            
            # Leer datos con pandas
            df = pd.read_sql(query, conn, params=params)
            
            # Generar archivo según formato
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if formato.lower() == "excel":
                output_path = ml.TEMP_PATH / f"anomalias_export_{timestamp}.xlsx"
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Anomalías', index=False)
                    
                    # Agregar hoja de resumen
                    resumen = pd.DataFrame({
                        'Métrica': [
                            'Total Anomalías',
                            'Países Únicos',
                            'Líneas Únicas',
                            'Score Promedio',
                            'Total Llamadas',
                            'Total Minutos'
                        ],
                        'Valor': [
                            len(df),
                            df['CODIGO_PAIS'].nunique(),
                            df['LINEA'].nunique(),
                            df['SCORE_ANOMALIA'].mean(),
                            df['N_LLAMADAS'].sum(),
                            df['N_MINUTOS'].sum()
                        ]
                    })
                    resumen.to_excel(writer, sheet_name='Resumen', index=False)
                    
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                filename = f"anomalias_{timestamp}.xlsx"
            else:
                output_path = ml.TEMP_PATH / f"anomalias_export_{timestamp}.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                content_type = "text/csv"
                filename = f"anomalias_{timestamp}.csv"
            
            return FileResponse(
                path=output_path,
                media_type=content_type,
                filename=filename
            )
            
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
    if not oracle.oracle_pool:
        return EstadoSistema(
            estado="oracle no disponible",
            modelo_cargado=bool(ml.modelo),
            ultimo_procesamiento=None,
            registros_procesados_hoy=0,
            anomalias_detectadas_hoy=0,
            espacio_disco_gb=0
        )

    try:
        # Calcular estadísticas del día desde Oracle
        hoy = datetime.now().date()
        
        with get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            # Registros procesados hoy
            cursor.execute("""
                SELECT COALESCE(SUM(TOTAL_REGISTROS), 0) as total_registros
                FROM LOTES_PROCESAMIENTO
                WHERE TRUNC(TIMESTAMP_INICIO) = TRUNC(SYSDATE)
                AND ESTADO = 'COMPLETADO'
            """)
            registros_hoy = cursor.fetchone()[0] or 0
            
            # Anomalías detectadas hoy
            cursor.execute("""
                SELECT COUNT(*) as total_anomalias
                FROM ALERTAS_FRAUDE
                WHERE TRUNC(FECHA_PROCESAMIENTO) = TRUNC(SYSDATE)
            """)
            anomalias_hoy = cursor.fetchone()[0] or 0
            
            # Último procesamiento
            cursor.execute("""
                SELECT MAX(TIMESTAMP_INICIO) as ultimo
                FROM LOTES_PROCESAMIENTO
                WHERE ESTADO = 'COMPLETADO'
            """)
            ultimo_row = cursor.fetchone()
            ultimo_procesamiento = ultimo_row[0] if ultimo_row and ultimo_row[0] else None

        # Espacio en disco
        try:
            stat = shutil.disk_usage(ml.TEMP_PATH)  # Usar TEMP_PATH en lugar de OUTPUT_PATH
            espacio_gb = stat.free / (1024**3)
        except:
            espacio_gb = 0

        return EstadoSistema(
            estado="saludable" if ml.modelo else "modelo no cargado",
            modelo_cargado=bool(ml.modelo),
            ultimo_procesamiento=ultimo_procesamiento,
            registros_procesados_hoy=int(registros_hoy),
            anomalias_detectadas_hoy=int(anomalias_hoy),
            espacio_disco_gb=round(espacio_gb, 2)
        )

    except Exception as e:
        logger.error(f"Error verificando salud: {e}")
        return EstadoSistema(
            estado="error",
            modelo_cargado=bool(ml.modelo),
            ultimo_procesamiento=None,
            registros_procesados_hoy=0,
            anomalias_detectadas_hoy=0,
            espacio_disco_gb=0
        )

@router.get("/dashboard/estadisticas-comparativas", tags=["Dashboard"])
async def obtener_estadisticas_comparativas():
    """
    Obtiene estadísticas del día actual vs día anterior para mostrar cambios porcentuales
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")
    
    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Registros procesados HOY
            cursor.execute("""
                SELECT COALESCE(SUM(TOTAL_REGISTROS), 0) as registros_hoy
                FROM LOTES_PROCESAMIENTO
                WHERE TRUNC(TIMESTAMP_INICIO) = TRUNC(SYSDATE)
                AND ESTADO = 'COMPLETADO'
            """)
            registros_hoy = cursor.fetchone()[0] or 0
            
            # 2. Anomalías detectadas HOY
            cursor.execute("""
                SELECT COUNT(*) as anomalias_hoy
                FROM ALERTAS_FRAUDE
                WHERE TRUNC(FECHA_PROCESAMIENTO) = TRUNC(SYSDATE)
            """)
            anomalias_hoy = cursor.fetchone()[0] or 0
            
            # 3. Registros procesados AYER
            cursor.execute("""
                SELECT COALESCE(SUM(TOTAL_REGISTROS), 0) as registros_ayer
                FROM LOTES_PROCESAMIENTO
                WHERE TRUNC(TIMESTAMP_INICIO) = TRUNC(SYSDATE-1)
                AND ESTADO = 'COMPLETADO'
            """)
            registros_ayer = cursor.fetchone()[0] or 0
            
            # 4. Anomalías detectadas AYER
            cursor.execute("""
                SELECT COUNT(*) as anomalias_ayer
                FROM ALERTAS_FRAUDE
                WHERE TRUNC(FECHA_PROCESAMIENTO) = TRUNC(SYSDATE-1)
            """)
            anomalias_ayer = cursor.fetchone()[0] or 0
            
            # 5. Calcular cambios porcentuales
            cambio_registros = 0
            if registros_ayer > 0:
                cambio_registros = ((registros_hoy - registros_ayer) / registros_ayer) * 100
            
            cambio_anomalias = 0
            if anomalias_ayer > 0:
                cambio_anomalias = ((anomalias_hoy - anomalias_ayer) / anomalias_ayer) * 100
            
            # 6. Calcular tasas de fraude
            tasa_hoy = 0
            if registros_hoy > 0:
                tasa_hoy = (anomalias_hoy / registros_hoy) * 100
                
            tasa_ayer = 0
            if registros_ayer > 0:
                tasa_ayer = (anomalias_ayer / registros_ayer) * 100
            
            cambio_tasa = tasa_hoy - tasa_ayer
            
            return {
                "registros_hoy": int(registros_hoy),
                "anomalias_hoy": int(anomalias_hoy),
                "registros_ayer": int(registros_ayer),
                "anomalias_ayer": int(anomalias_ayer),
                "tasa_fraude_hoy": round(float(tasa_hoy), 2),
                "tasa_fraude_ayer": round(float(tasa_ayer), 2),
                "cambio_registros": round(float(cambio_registros), 1),
                "cambio_anomalias": round(float(cambio_anomalias), 1),
                "cambio_tasa": round(float(cambio_tasa), 1)
            }
            
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas comparativas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                    ID_ACTUALIZACION,
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
                FROM ACTUALIZACION_DIARIA_UMBRAL
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
async def detalle_aprendizaje(ID_ACTUALIZACION: int):
    """
    Obtiene el detalle completo de un aprendizaje incremental específico
    """
    if not oracle.oracle_pool:
        raise HTTPException(status_code=503, detail="Oracle no disponible")

    try:
        with get_oracle_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM ACTUALIZACION_DIARIA_UMBRAL
                WHERE ID_ACTUALIZACION = :id
            """, [ID_ACTUALIZACION])

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
                FROM ACTUALIZACION_DIARIA_UMBRAL
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
                FROM ACTUALIZACION_DIARIA_UMBRAL
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
                FROM ACTUALIZACION_DIARIA_UMBRAL
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

@router.get("/analisis/tendencias")
async def obtener_tendencias_fraude(
    dias: int = Query(default=7, description="Número de días a analizar")
):
    """Obtiene tendencias de fraude para los últimos N días"""
    try:
        with oracle.get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT 
                TRUNC(FECHA_PROCESAMIENTO) as FECHA,
                COUNT(DISTINCT LOTE_PROCESAMIENTO) as LOTES_PROCESADOS,
                COUNT(*) as TOTAL_ALERTAS,
                COUNT(DISTINCT CODIGO_PAIS) as PAISES_AFECTADOS,
                COUNT(DISTINCT LINEA) as LINEAS_AFECTADAS,
                ROUND(AVG(SCORE_ANOMALIA), 4) as SCORE_PROMEDIO,
                SUM(N_LLAMADAS) as TOTAL_LLAMADAS,
                SUM(N_MINUTOS) as TOTAL_MINUTOS
            FROM ALERTAS_FRAUDE
            WHERE FECHA_PROCESAMIENTO >= SYSDATE - :dias
            GROUP BY TRUNC(FECHA_PROCESAMIENTO)
            ORDER BY FECHA DESC
            """
            
            cursor.execute(query, dias=dias)
            columns = [col[0] for col in cursor.description]
            resultados = []
            
            for row in cursor:
                resultado = dict(zip(columns, row))
                resultado['FECHA'] = resultado['FECHA'].strftime('%Y-%m-%d')
                resultados.append(resultado)
                
            # Calcular tendencias día a día
            tendencias = []
            for i, dia in enumerate(resultados):
                tendencia = {
                    "fecha": dia['FECHA'],
                    "anomalias": dia['TOTAL_ALERTAS'],
                    "lotes": dia['LOTES_PROCESADOS'],
                    "score_promedio": float(dia['SCORE_PROMEDIO']),
                    "paises_afectados": dia['PAISES_AFECTADOS'],
                    "lineas_afectadas": dia['LINEAS_AFECTADAS']
                }
                
                # Obtener el total de registros procesados ese día
                query_total = """
                SELECT SUM(TOTAL_REGISTROS) as TOTAL
                FROM LOTES_PROCESAMIENTO
                WHERE TRUNC(TIMESTAMP_INICIO) = TO_DATE(:fecha, 'YYYY-MM-DD')
                """
                cursor.execute(query_total, fecha=dia['FECHA'])
                total_row = cursor.fetchone()
                total = total_row[0] if total_row and total_row[0] else 0
                
                tendencia["total"] = total
                tendencia["tasa"] = round((dia['TOTAL_ALERTAS'] / total * 100) if total > 0 else 0, 2)
                
                tendencias.append(tendencia)
                
        return {"tendencias": tendencias}
        
    except Exception as e:
        logger.error(f"Error obteniendo tendencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alertas/estadisticas")
async def obtener_estadisticas_alertas(
    fecha_inicio: Optional[str] = Query(None, description="Fecha inicio YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha fin YYYY-MM-DD")
):
    """Obtiene estadísticas detalladas de alertas para el dashboard"""
    try:
        with oracle.get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            # Construir filtro de fechas
            where_clause = "WHERE 1=1"
            params = {}
            
            if fecha_inicio:
                where_clause += " AND FECHA_PROCESAMIENTO >= TO_DATE(:fecha_inicio, 'YYYY-MM-DD')"
                params['fecha_inicio'] = fecha_inicio
            if fecha_fin:
                where_clause += " AND FECHA_PROCESAMIENTO <= TO_DATE(:fecha_fin, 'YYYY-MM-DD') + 1"
                params['fecha_fin'] = fecha_fin
            
            # Estadísticas generales
            query_general = f"""
            SELECT 
                COUNT(*) as TOTAL_ALERTAS,
                COUNT(DISTINCT CODIGO_PAIS) as PAISES_UNICOS,
                COUNT(DISTINCT LINEA) as LINEAS_UNICAS,
                ROUND(AVG(SCORE_ANOMALIA), 4) as SCORE_PROMEDIO,
                MAX(SCORE_ANOMALIA) as SCORE_MAXIMO,
                SUM(N_LLAMADAS) as TOTAL_LLAMADAS_FRAUDE,
                SUM(N_MINUTOS) as TOTAL_MINUTOS_FRAUDE
            FROM ALERTAS_FRAUDE
            {where_clause}
            """
            
            cursor.execute(query_general, **params)
            estadisticas_generales = dict(zip(
                [col[0].lower() for col in cursor.description],
                cursor.fetchone()
            ))
            
            # Top países con más fraudes
            query_paises = f"""
            SELECT 
                CODIGO_PAIS,
                COUNT(*) as CANTIDAD,
                ROUND(AVG(SCORE_ANOMALIA), 4) as SCORE_PROMEDIO,
                SUM(N_LLAMADAS) as TOTAL_LLAMADAS,
                SUM(N_MINUTOS) as TOTAL_MINUTOS
            FROM ALERTAS_FRAUDE
            {where_clause}
            GROUP BY CODIGO_PAIS
            ORDER BY CANTIDAD DESC
            FETCH FIRST 10 ROWS ONLY
            """
            
            cursor.execute(query_paises, **params)
            top_paises = []
            for row in cursor:
                top_paises.append({
                    "codigo_pais": row[0],
                    "cantidad": row[1],
                    "score_promedio": float(row[2]),
                    "total_llamadas": row[3],
                    "total_minutos": float(row[4])
                })
            
            # Distribución por tipo de anomalía
            query_tipos = f"""
            SELECT 
                TIPO_ANOMALIA,
                COUNT(*) as CANTIDAD,
                ROUND(AVG(SCORE_ANOMALIA), 4) as SCORE_PROMEDIO
            FROM ALERTAS_FRAUDE
            {where_clause}
            GROUP BY TIPO_ANOMALIA
            ORDER BY CANTIDAD DESC
            """
            
            cursor.execute(query_tipos, **params)
            distribucion_tipos = []
            for row in cursor:
                distribucion_tipos.append({
                    "tipo": row[0],
                    "cantidad": row[1],
                    "score_promedio": float(row[2])
                })
            
            # Evolución por hora del día (últimas 24 horas)
            query_horas = """
            SELECT 
                TO_CHAR(FECHA_PROCESAMIENTO, 'HH24') as HORA,
                COUNT(*) as CANTIDAD
            FROM ALERTAS_FRAUDE
            WHERE FECHA_PROCESAMIENTO >= SYSDATE - 1
            GROUP BY TO_CHAR(FECHA_PROCESAMIENTO, 'HH24')
            ORDER BY HORA
            """
            
            cursor.execute(query_horas)
            evolucion_24h = []
            for row in cursor:
                evolucion_24h.append({
                    "hora": f"{row[0]}:00",
                    "cantidad": row[1]
                })
            
            return {
                "estadisticas_generales": estadisticas_generales,
                "top_paises": top_paises,
                "distribucion_tipos": distribucion_tipos,
                "evolucion_24h": evolucion_24h
            }
            
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analisis/exportar-anomalias")
async def exportar_anomalias(
    formato: str = Query("csv", description="Formato de exportación: csv o excel"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha inicio YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha fin YYYY-MM-DD"),
    limite: int = Query(10000, description="Límite de registros")
):
    """Exporta las anomalías detectadas en formato CSV o Excel"""
    try:
        with oracle.get_oracle_connection() as conn:
            # Construir query con filtros
            where_clause = "WHERE 1=1"
            params = {"limite": limite}
            
            if fecha_inicio:
                where_clause += " AND FECHA_PROCESAMIENTO >= TO_DATE(:fecha_inicio, 'YYYY-MM-DD')"
                params['fecha_inicio'] = fecha_inicio
            if fecha_fin:
                where_clause += " AND FECHA_PROCESAMIENTO <= TO_DATE(:fecha_fin, 'YYYY-MM-DD') + 1"
                params['fecha_fin'] = fecha_fin
                
            query = f"""
            SELECT 
                ID_ALERTA,
                TO_CHAR(FECHA_PROCESAMIENTO, 'YYYY-MM-DD HH24:MI:SS') as FECHA_PROCESAMIENTO,
                FECHA_REGISTRO,
                CODIGO_PAIS,
                LINEA,
                N_LLAMADAS,
                N_MINUTOS,
                N_DESTINOS,
                SCORE_ANOMALIA,
                UMBRAL,
                TIPO_ANOMALIA,
                TIPO_CONTEXTO,
                RAZON_DECISION,
                ARCHIVO_ORIGEN,
                LOTE_PROCESAMIENTO
            FROM ALERTAS_FRAUDE
            {where_clause}
            ORDER BY FECHA_PROCESAMIENTO DESC
            FETCH FIRST :limite ROWS ONLY
            """
            
            # Leer datos con pandas
            df = pd.read_sql(query, conn, params=params)
            
            # Generar archivo según formato
            if formato.lower() == "excel":
                output_path = ml.TEMP_PATH / f"anomalias_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Anomalías', index=False)
                    
                    # Agregar hoja de resumen
                    resumen = pd.DataFrame({
                        'Métrica': [
                            'Total Anomalías',
                            'Países Únicos',
                            'Líneas Únicas',
                            'Score Promedio',
                            'Total Llamadas',
                            'Total Minutos'
                        ],
                        'Valor': [
                            len(df),
                            df['CODIGO_PAIS'].nunique(),
                            df['LINEA'].nunique(),
                            df['SCORE_ANOMALIA'].mean(),
                            df['N_LLAMADAS'].sum(),
                            df['N_MINUTOS'].sum()
                        ]
                    })
                    resumen.to_excel(writer, sheet_name='Resumen', index=False)
                    
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                filename = f"anomalias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            else:
                output_path = ml.TEMP_PATH / f"anomalias_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                content_type = "text/csv"
                filename = f"anomalias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            return FileResponse(
                path=output_path,
                media_type=content_type,
                filename=filename
            )
            
    except Exception as e:
        logger.error(f"Error exportando anomalías: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/alertas")
async def websocket_alertas(websocket: WebSocket):
    """WebSocket para alertas en tiempo real"""
    await websocket.accept()
    try:
        while True:
            # Simular envío de alertas cada 5 segundos
            await asyncio.sleep(5)
            
            # Obtener última alerta
            with oracle.get_oracle_connection() as conn:
                cursor = conn.cursor()
                query = """
                SELECT 
                    ID_ALERTA,
                    TO_CHAR(FECHA_PROCESAMIENTO, 'YYYY-MM-DD HH24:MI:SS') as FECHA,
                    CODIGO_PAIS,
                    LINEA,
                    TIPO_ANOMALIA,
                    SCORE_ANOMALIA
                FROM ALERTAS_FRAUDE
                WHERE FECHA_PROCESAMIENTO >= SYSDATE - INTERVAL '1' MINUTE
                ORDER BY FECHA_PROCESAMIENTO DESC
                FETCH FIRST 1 ROW ONLY
                """
                cursor.execute(query)
                row = cursor.fetchone()
                
                if row:
                    alerta = {
                        "id": row[0],
                        "fecha": row[1],
                        "codigo_pais": row[2],
                        "linea": row[3],
                        "tipo": row[4],
                        "score": float(row[5]),
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(alerta)
                    
    except WebSocketDisconnect:
        logger.info("Cliente desconectado del WebSocket")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        await websocket.close()


# Endpoint para métricas en tiempo real
@router.get("/metricas/tiempo-real")
async def obtener_metricas_tiempo_real():
    """Obtiene métricas en tiempo real para el dashboard"""
    try:
        with oracle.get_oracle_connection() as conn:
            cursor = conn.cursor()
            
            # Métricas de las últimas 24 horas
            query = """
            SELECT 
                COUNT(*) as ALERTAS_24H,
                COUNT(DISTINCT LOTE_PROCESAMIENTO) as LOTES_24H,
                COUNT(CASE WHEN FECHA_PROCESAMIENTO >= SYSDATE - 1/24 THEN 1 END) as ALERTAS_1H,
                COUNT(CASE WHEN TIPO_ANOMALIA = 'SPRAY_CALLING' THEN 1 END) as SPRAY_CALLING_24H,
                COUNT(CASE WHEN TIPO_ANOMALIA = 'MINUTOS_EXTREMOS' THEN 1 END) as MINUTOS_EXTREMOS_24H,
                ROUND(AVG(SCORE_ANOMALIA), 4) as SCORE_PROMEDIO_24H
            FROM ALERTAS_FRAUDE
            WHERE FECHA_PROCESAMIENTO >= SYSDATE - 1
            """
            
            cursor.execute(query)
            metricas = dict(zip(
                [col[0].lower() for col in cursor.description],
                cursor.fetchone()
            ))
            
            # Tasa de crecimiento
            query_anterior = """
            SELECT COUNT(*) as ALERTAS_ANTERIOR
            FROM ALERTAS_FRAUDE
            WHERE FECHA_PROCESAMIENTO >= SYSDATE - 2
              AND FECHA_PROCESAMIENTO < SYSDATE - 1
            """
            
            cursor.execute(query_anterior)
            alertas_anterior = cursor.fetchone()[0]
            
            tasa_crecimiento = 0
            if alertas_anterior > 0:
                tasa_crecimiento = round(
                    ((metricas['alertas_24h'] - alertas_anterior) / alertas_anterior) * 100, 
                    2
                )
            
            metricas['tasa_crecimiento_24h'] = tasa_crecimiento
            
            return metricas
            
    except Exception as e:
        logger.error(f"Error obteniendo métricas en tiempo real: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ## 11. CONFIGURACIÓN PARA EJECUTAR LA API

