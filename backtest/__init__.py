"""
Módulo de Backtest - Simulación de estrategias de trading
==========================================================
Contiene el motor principal de backtest y funciones de métricas.
"""

from .engine import run_backtest, _get_indicator
from .metrics import calcular_drawdown_maximo, calcular_score

__all__ = [
    'run_backtest',
    '_get_indicator',
    'calcular_drawdown_maximo',
    'calcular_score'
]