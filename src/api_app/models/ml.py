import pickle
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .schemas import TipoAnomalia, Criticidad

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_PATH = BASE_DIR / "Modelos"
TEMP_PATH = BASE_DIR / "Temp_API"
LOG_PATH = BASE_DIR / "Logs_API"

for path in [TEMP_PATH, LOG_PATH]:
    path.mkdir(parents=True, exist_ok=True)

modelo = None
scaler = None
config = None
stats_dict = None
contexto_historico = None
umbral_global = None
parametros_features = None

executor = ThreadPoolExecutor(max_workers=4)


def cargar_modelo():
    """Carga el modelo y configuración desde disco"""
    global modelo, scaler, config, stats_dict, contexto_historico, umbral_global, parametros_features
    try:
        with open(MODELS_PATH / "modelo_general.pkl", "rb") as f:
            modelo = pickle.load(f)
        with open(MODELS_PATH / "scaler_general.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODELS_PATH / "config_modelo_general.pkl", "rb") as f:
            config = pickle.load(f)
        stats_dict = config["stats_por_pais"]
        contexto_historico = config.get("contexto_historico", {})
        umbral_global = config["umbral_global"]
        parametros_features = config["parametros_features"]
        logger.info("Modelo cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error al cargar modelo: {e}")
        return False


def crear_features_contextualizadas_mejorada(row, stats_pais_dict):
    """Crea features para el modelo"""
    pais = row["CODIGODEPAIS"]
    llamadas = row["N_LLAMADAS"]
    minutos = row["N_MINUTOS"]
    destinos = row["N_DESTINOS"]

    PESO_MINUTOS_NORMAL = parametros_features["peso_minutos_normal"]
    PESO_MINUTOS_EXTREMOS = parametros_features["peso_minutos_extremos"]
    UMBRAL_MINUTOS_EXTREMOS = parametros_features["umbral_minutos_extremos"]
    PESO_DESTINOS = parametros_features["peso_destinos"]
    PESO_SPRAY_RATIO = parametros_features["peso_spray_ratio"]

    if pais in stats_pais_dict:
        pais_stats = stats_pais_dict[pais]
        categoria = pais_stats["CATEGORIA"]
        llamadas_norm = min(llamadas / max(pais_stats["LLAMADAS_P95"], 1), 1.5)
        destinos_norm = min(destinos / max(pais_stats["DESTINOS_P95"], 1), 1.5)
        minutos_p90 = pais_stats.get("MINUTOS_P90", pais_stats["MINUTOS_P95"] * 0.9)
        if minutos >= UMBRAL_MINUTOS_EXTREMOS:
            minutos_norm = min(minutos / max(minutos_p90, 1), 3.0)
            peso_minutos = PESO_MINUTOS_EXTREMOS
        else:
            minutos_norm = min(np.log1p(minutos) / np.log1p(max(minutos_p90, 1)), 1.2)
            peso_minutos = PESO_MINUTOS_NORMAL
    else:
        categoria = "Muy_Bajo"
        llamadas_norm = min(llamadas / 10, 2.0)
        destinos_norm = min(destinos / 5, 2.0)
        if minutos >= UMBRAL_MINUTOS_EXTREMOS:
            minutos_norm = min(minutos / 50, 3.0)
            peso_minutos = PESO_MINUTOS_EXTREMOS * 1.2
        else:
            minutos_norm = min(np.log1p(minutos) / np.log1p(60), 1.2)
            peso_minutos = PESO_MINUTOS_NORMAL

    features = {
        "llamadas_norm": llamadas_norm * 0.8,
        "destinos_norm": destinos_norm * PESO_DESTINOS,
        "minutos_norm": minutos_norm * peso_minutos,
        "diversidad_destinos": min(destinos / max(llamadas, 1), 1.0),
        "spray_ratio": min(destinos / max(llamadas, 1) * PESO_SPRAY_RATIO, 1.0) if destinos >= 5 else 0,
        "minutos_extremos": 1.0 if minutos >= UMBRAL_MINUTOS_EXTREMOS else 0.0,
        "minutos_sospechosos": min((minutos - 200) / 300, 1.0) if minutos > 200 else 0.0,
        "patron_spray_fuerte": 1.0 if (destinos >= 10 and llamadas >= 20) else 0.0,
        "patron_spray_medio": 0.5 if (destinos >= 6 and llamadas >= 12) else 0.0,
        "alta_diversidad": min(destinos / 12, 1) if destinos >= 5 else 0,
        "volumen_llamadas_alto": min((llamadas - 30) / 50, 1) if llamadas > 30 else 0,
        "volumen_destinos_alto": min((destinos - 10) / 20, 1) if destinos > 10 else 0,
        "llamadas_por_destino": min(llamadas / max(destinos, 1) / 5, 1),
        "eficiencia_destinos": min(destinos / max(llamadas * 0.5, 1), 1),
        "factor_pais_bajo": 1.5 if categoria in ["Muy_Bajo", "Bajo"] else 1.0,
        "factor_pais_alto": 0.9 if categoria in ["Alto", "Medio"] else 1.0,
    }
    return features


def calcular_criticidad(
    score,
    umbral,
    llamadas,
    minutos,
    destinos,
    pais=None,
    divisores=None,
):
    """Calcula la criticidad de un registro.

    Los divisores para normalizar las variables pueden provenir de:
    1. Un diccionario ``divisores`` pasado explícitamente.
    2. Las estadísticas del país en ``stats_dict`` utilizando percentiles.
    3. Parámetros configurables en ``parametros_features``.
    """

    ratio_score = max(score - umbral, 0) / umbral if umbral else 0

    if divisores is not None:
        divisor_llamadas = divisores.get("llamadas", 100)
        divisor_minutos = divisores.get("minutos", 300)
        divisor_destinos = divisores.get("destinos", 50)
    elif pais is not None and stats_dict and pais in stats_dict:
        stats = stats_dict[pais]
        divisor_llamadas = stats.get("LLAMADAS_P95", 100)
        divisor_minutos = stats.get("MINUTOS_P95", 300)
        divisor_destinos = stats.get("DESTINOS_P95", 50)
    else:
        divisor_llamadas = (
            parametros_features.get("divisor_llamadas", 100)
            if parametros_features
            else 100
        )
        divisor_minutos = (
            parametros_features.get("divisor_minutos", 300)
            if parametros_features
            else 300
        )
        divisor_destinos = (
            parametros_features.get("divisor_destinos", 50)
            if parametros_features
            else 50
        )

    factor_llamadas = llamadas / divisor_llamadas
    factor_minutos = minutos / divisor_minutos
    factor_destinos = destinos / divisor_destinos

    riesgo = (
        0.4 * ratio_score
        + 0.3 * factor_llamadas
        + 0.2 * factor_minutos
        + 0.1 * factor_destinos
    )
    if riesgo > 3.0:
        return Criticidad.CRITICA
    elif riesgo > 1.5:
        return Criticidad.ALTA
    elif riesgo > 1.0:
        return Criticidad.MEDIA
    else:
        return Criticidad.BAJA


def predecir_anomalia_individual(row_dict):
    """Realiza predicción para un registro individual"""
    if not modelo:
        raise ValueError("Modelo no cargado")

    pais = int(row_dict["CODIGODEPAIS"]) if isinstance(row_dict["CODIGODEPAIS"], str) else row_dict["CODIGODEPAIS"]
    llamadas = row_dict["N_LLAMADAS"]
    minutos = row_dict["N_MINUTOS"]
    destinos = row_dict["N_DESTINOS"]
    row_dict["CODIGODEPAIS"] = pais

    features = crear_features_contextualizadas_mejorada(row_dict, stats_dict)
    features_scaled = scaler.transform_one(features)
    score = modelo.score_one(features_scaled)
    es_anomalia_base = score > umbral_global

    if es_anomalia_base:
        if minutos >= parametros_features["umbral_minutos_extremos"]:
            es_anomalia_final = True
            razon = f"Minutos extremos ({minutos:.1f} min)"
            tipo_anomalia = TipoAnomalia.MINUTOS_EXTREMOS
        elif destinos >= 6 and llamadas >= 12:
            es_anomalia_final = True
            razon = "Patrón de spray calling confirmado"
            tipo_anomalia = TipoAnomalia.SPRAY_CALLING
        elif llamadas > 50 or destinos > 15:
            es_anomalia_final = True
            razon = "Volumen excepcionalmente alto"
            tipo_anomalia = TipoAnomalia.VOLUMEN_ALTO
        elif pais not in stats_dict or stats_dict.get(pais, {}).get("CATEGORIA") in ["Muy_Bajo", "Bajo"]:
            if destinos >= 4 and llamadas >= 8:
                es_anomalia_final = True
                razon = "Actividad sospechosa en país de bajo tráfico"
                tipo_anomalia = TipoAnomalia.PAIS_BAJO_TRAFICO
            else:
                es_anomalia_final = False
                razon = "Actividad baja en país de bajo tráfico"
                tipo_anomalia = TipoAnomalia.NO_ANOMALIA
        else:
            es_anomalia_final = False
            razon = "No cumple criterios de confirmación"
            tipo_anomalia = TipoAnomalia.NO_ANOMALIA
    else:
        es_anomalia_final = False
        razon = "Score bajo umbral"
        tipo_anomalia = TipoAnomalia.NO_ANOMALIA
    criticidad = calcular_criticidad(
        score, umbral_global, llamadas, minutos, destinos, pais=pais
    )

    if contexto_historico and pais in contexto_historico:
        tipo_contexto = contexto_historico[pais]
    elif pais in stats_dict:
        tipo_contexto = stats_dict[pais]["CATEGORIA"]
    else:
        tipo_contexto = "Muy_Bajo"

    return {
        "score": score,
        "umbral": umbral_global,
        "es_anomalia": es_anomalia_final,
        "tipo_anomalia": tipo_anomalia,
        "criticidad": criticidad,
        "tipo_contexto": tipo_contexto,
        "razon_decision": razon,
    }

