"""
Configuración Global del Optimizador
====================================
Contiene constantes y configuraciones compartidas entre módulos.
"""




# ============================================================================
# CONFIGURACIÓN DE FEATURES (por defecto) (antes en optimizador)
# ============================================================================


FEATURES = {
    "enable_long_trades": True,      # Habilitar operaciones largas
    "enable_short_trades": True,     # Habilitar operaciones cortas
    "use_rsi_long": False,           # Filtro RSI para longs
    "use_rsi_short": False,          # Filtro RSI para shorts
    "use_adx_filter": False,         # Filtro de tendencia ADX
    "enable_high_condition": True,   # Condición de nuevo máximo
    "enable_low_condition": True,    # Condición de nuevo mínimo
    "use_validation_window": True,   # Ventana de validación post-cruce
    "use_htf_filter": False,         # Filtro de timeframe superior
    "use_stop_loss": False,          # Stop loss porcentual
    "activar_stop_be": False,        # Break even dinámico
    "enable_cooldown": False,        # Período de enfriamiento tras pérdidas
    "enable_reentry": True,          # Permitir reentradas
    "enable_post_crossover_entry": False,  # Entradas post-cruce
    "use_take_profit_long": False,   # Take profit para longs
    "use_take_profit_short": False,  # Take profit para shorts
}





# ============================================================================
# CONSTANTES DEL BACKTEST (antes en optimizador)
# ============================================================================


CONSTANTS = {
    "commission_pct": 0.075 / 100.0,  # Comisión (0.075%)
    "initial_capital": 2000.0,        # Capital inicial en USD
    "risk_pct_per_trade": 0.30,       # Riesgo por operación (30% del capital)
    "htf_tf": "1d",                   # Timeframe superior por defecto
    "htf_type": "SMA",                # Tipo de MA para HTF
}



# ============================================================================
# CONFIGURACIÓN DE TIMEFRAMES (antes en optimizador)
# ============================================================================


TIMEFRAME_CONFIG = {
    "1m": {"interval_min": 1, "pct_threshold": 0.02, "intraday": True},
    "5m": {"interval_min": 5, "pct_threshold": 0.04, "intraday": True},
    "15m": {"interval_min": 15, "pct_threshold": 0.06, "intraday": True},
    "30m": {"interval_min": 30, "pct_threshold": 0.08, "intraday": True},
    "1h": {"interval_min": 60, "pct_threshold": 0.10, "intraday": True},
    "2h": {"interval_min": 120, "pct_threshold": 0.12, "intraday": True},
    "4h": {"interval_min": 240, "pct_threshold": 0.15, "intraday": True},
    "1d": {"interval_min": 1440, "pct_threshold": None, "intraday": False},
}




# ============================================================================
# RANGOS POR DEFECTO (para GUI) (antes en GUI)
# ============================================================================


DEFAULT_RANGES = {
    "rsi_length": (2, 50),
    "rsi_min": (50.0, 80.0),
    "rsi_max": (5.0, 50.0),
    "adx_length": (2, 50),
    "adx_threshold": (5.0, 70.0),
    "lookback": (1, 15),
    "validation_window": (1, 15),
    "htf_length": (10, 60),
    "stop_loss": (0.3, 10.0),
    "be": (1, 10),
    "tp_long": (0.3, 99.0),
    "tp_short": (0.3, 99.0),
    "mls": (1, 4),
    "cooldown": (10, 300),
    "re": (1, 4),
    "postre": (1, 4),
}