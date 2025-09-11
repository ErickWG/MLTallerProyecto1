import os
import sys

# Ensure src is on the path for imports
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "..", "src"))

import api_app.models.ml as ml
from api_app.models.schemas import Criticidad

# Configure stats_dict to provide percentile divisors
ml.stats_dict = {
    1: {"LLAMADAS_P95": 100, "MINUTOS_P95": 300, "DESTINOS_P95": 50}
}
ml.parametros_features = {}

def test_calcular_criticidad_baja():
    result = ml.calcular_criticidad(1.0, 1.0, 50, 150, 25, pais=1)
    assert result == Criticidad.BAJA

def test_calcular_criticidad_media():
    result = ml.calcular_criticidad(1.5, 1.0, 150, 450, 75, pais=1)
    assert result == Criticidad.MEDIA

def test_calcular_criticidad_alta():
    result = ml.calcular_criticidad(2.0, 1.0, 200, 600, 100, pais=1)
    assert result == Criticidad.ALTA

def test_calcular_criticidad_critica():
    result = ml.calcular_criticidad(6.0, 1.0, 300, 900, 150, pais=1)
    assert result == Criticidad.CRITICA


def test_calcular_criticidad_critica_por_destinos():
    """High destination count should trigger CRITICA even with low score."""
    result = ml.calcular_criticidad(1.0, 1.0, 150, 200, 350, pais=1)
    assert result == Criticidad.CRITICA
