"""
CONFIGURACIÓN DE OPTIMIZACIÓN AUTOMÁTICA
========================================
Define los valores por defecto para las 3 fases de optimización automática.
El usuario puede modificar estos valores desde el GUI.
"""

# ============================================================================
# CONFIGURACIÓN DE FASES (trials y modos)
# ============================================================================

CONFIG_FASES = {
    "fase_1": {
        "nombre": "Exploración Rápida",
        "trials": 2000,
        "modo": "paralelo",           # "paralelo" o "serie"
        "multi_run": False,
        "runs": 1,
        "rangos_acotados": False
    },
    "fase_2": {
        "nombre": "Refinamiento",
        "trials": 1500,
        "modo": "serie",
        "multi_run": False,
        "runs": 1,
        "rangos_acotados": True
    },
    "fase_3": {
        "nombre": "Validación",
        "trials_por_corrida": 800,
        "modo": "serie",
        "corridas": 5,
        "multi_run": True,
        "rangos_acotados": False       # Usa rangos originales para validar
    }
}

# ============================================================================
# CONFIGURACIÓN DE CONVERGENCIA (solo para Fase 1)
# ============================================================================

CONFIG_CONVERGENCIA = {
    "activar": True,                   # Si True, detiene Fase 1 si converge
    "ventana": 75,                     # Número de trials a mirar hacia atrás
    "tolerancia": 0.002,               # 0.2% - mejora mínima para seguir
    "trials_minimos": 400,             # No evaluar convergencia antes de esto
    "mejor_score_minimo": 0.85         # Si supera esto, también converge
}

# ============================================================================
# CONFIGURACIÓN DE MÉTRICAS POR FASE
# ============================================================================
# Cada fase tiene sus propios pesos para las métricas.
# El usuario puede modificar estos valores desde el GUI.
# ============================================================================

CONFIG_METRICAS = {
    "fase_1": {
        "use_pf": True, "peso_pf": 60.0,
        "use_winrate": True, "peso_winrate": 40.0,
        "use_drawdown": False, "peso_drawdown": 0.0,
        "use_n_trades": False, "peso_n_trades": 0.0,
        "min_trades": 15
    },
    "fase_2": {
        "use_pf": True, "peso_pf": 40.0,
        "use_winrate": True, "peso_winrate": 30.0,
        "use_drawdown": True, "peso_drawdown": 20.0,
        "use_n_trades": True, "peso_n_trades": 10.0,
        "min_trades": 20
    },
    "fase_3": {
        "use_pf": True, "peso_pf": 35.0,
        "use_winrate": True, "peso_winrate": 30.0,
        "use_drawdown": True, "peso_drawdown": 35.0,
        "use_n_trades": False, "peso_n_trades": 0.0,
        "min_trades": 30
    }
}

# ============================================================================
# VALORES POR DEFECTO PARA RESET (GUI)
# ============================================================================

DEFAULTS_AUTO = {
    # ===== CONVERGENCIA =====
    "auto_activar_convergencia": True,
    "auto_ventana": "75",
    "auto_tolerancia": "0.002",
    "auto_trials_minimos": "400",
    "auto_mejor_score_minimo": "0.85",
    
    # Fase 1
    "auto_fase1_trials": "2000",
    "auto_fase1_modo_serie": False,
    "auto_fase1_modo_paralelo": True,  # Paralelo por defecto
    "auto_fase1_multi_run": False,
    "auto_fase1_runs": "1",
    
    # Fase 2
    "auto_fase2_trials": "1500",
    "auto_fase2_modo_serie": True,   # Serie por defecto
    "auto_fase2_modo_paralelo": False,
    "auto_fase2_multi_run": False,
    "auto_fase2_runs": "1",
    
    # Fase 3
    "auto_fase3_trials_x_corrida": "800",
    "auto_fase3_modo_serie": True,   # Serie por defecto
    "auto_fase3_modo_paralelo": False,
    "auto_fase3_multi_run": True,
    "auto_fase3_runs": "5",
    
    # ===== MÉTRICAS FASE 1 =====
    "auto_fase1_pf": "60",
    "auto_fase1_winrate": "40",
    "auto_fase1_drawdown": "0",
    "auto_fase1_n_trades": "0",
    "auto_fase1_min_trades": "15",
    "auto_fase1_use_pf": True,
    "auto_fase1_use_winrate": True,
    "auto_fase1_use_drawdown": False,
    "auto_fase1_use_n_trades": False,
    
    # ===== MÉTRICAS FASE 2 =====
    "auto_fase2_pf": "40",
    "auto_fase2_winrate": "30",
    "auto_fase2_drawdown": "20",
    "auto_fase2_n_trades": "10",
    "auto_fase2_min_trades": "20",
    "auto_fase2_use_pf": True,
    "auto_fase2_use_winrate": True,
    "auto_fase2_use_drawdown": True,
    "auto_fase2_use_n_trades": True,
    
    # ===== MÉTRICAS FASE 3 =====
    "auto_fase3_pf": "35",
    "auto_fase3_winrate": "30",
    "auto_fase3_drawdown": "35",
    "auto_fase3_n_trades": "0",
    "auto_fase3_min_trades": "30",
    "auto_fase3_use_pf": True,
    "auto_fase3_use_winrate": True,
    "auto_fase3_use_drawdown": True,
    "auto_fase3_use_n_trades": False,
}