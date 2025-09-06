from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime

class TipoAnomalia(str, Enum):
    MINUTOS_EXTREMOS = "MINUTOS_EXTREMOS"
    SPRAY_CALLING = "SPRAY_CALLING"
    VOLUMEN_ALTO = "VOLUMEN_ALTO"
    PAIS_BAJO_TRAFICO = "PAIS_BAJO_TRAFICO"
    NO_ANOMALIA = "NO_ANOMALIA"


class Criticidad(str, Enum):
    BAJA = "BAJA"
    MEDIA = "MEDIA"
    ALTA = "ALTA"
    CRITICA = "CRITICA"


class RegistroTelefonico(BaseModel):
    fecha: str = Field(..., description="Fecha en formato DD/MM/YYYY")
    codigodepais: int = Field(..., description="Código del país como número entero")
    linea: str = Field(..., description="Número de línea telefónica")
    n_llamadas: int = Field(..., ge=0, description="Número de llamadas")
    n_minutos: float = Field(..., ge=0, description="Minutos totales")
    n_destinos: int = Field(..., ge=0, description="Número de destinos únicos")


class ResultadoScoring(BaseModel):
    fecha: str
    codigodepais: int
    linea: str
    n_llamadas: int
    n_minutos: float
    n_destinos: int
    score_anomalia: float
    umbral: float
    es_anomalia: bool
    tipo_anomalia: TipoAnomalia
    criticidad: Criticidad
    tipo_contexto: str
    razon_decision: str
    timestamp_procesamiento: datetime


class EstadisticasModelo(BaseModel):
    fecha_entrenamiento: str
    fecha_ultima_actualizacion: Optional[str]
    umbral_global: float
    n_trees: int
    tree_height: int
    registros_entrenamiento: int
    paises_entrenamiento: int
    paises_conocidos: int
    version: str = "1.0"


class ConfiguracionUmbral(BaseModel):
    nuevo_umbral: float = Field(..., gt=0, le=1, description="Nuevo umbral entre 0 y 1")
    razon: str = Field(..., description="Razón del cambio")


class EstadoSistema(BaseModel):
    estado: str
    modelo_cargado: bool
    ultimo_procesamiento: Optional[datetime]
    registros_procesados_hoy: int
    anomalias_detectadas_hoy: int
    espacio_disco_gb: float
    version_api: str = "1.0"

