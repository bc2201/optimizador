"""
Prueba comparativa: Modo Serie vs Paralelo
Mismos parámetros, misma configuración
"""

import pandas as pd
import numpy as np
from optimization.auto_optimizer import OptimizadorAutomatico
from config import CONSTANTS

# ============================================================
# CONFIGURACIÓN COMÚN PARA AMBAS PRUEBAS
# ============================================================

# Cargar datos (usa los mismos que en tu GUI)
# Ajusta la ruta según donde tengas los datos de SPY
try:
    # Intentar cargar datos desde CSV si existe
    df = pd.read_csv("datos/SPY_1d.csv", index_col=0, parse_dates=True)
    print(f"[INFO] Datos cargados desde CSV: {len(df)} velas")
except:
    # Si no, usar datos sintéticos para la prueba
    print("[INFO] Generando datos sintéticos para la prueba...")
    dates = pd.date_range('2020-01-01', periods=1000, freq='1D')
    prices = 300 + np.cumsum(np.random.randn(1000) * 2)
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    print(f"[INFO] Datos sintéticos generados: {len(df)} velas")

# Configuración base (similar a la que usaste)
config_base = {
    "tipos_ma": ["EMA", "SMA"],
    "ma1_min": 5, "ma1_max": 20,
    "ma2_min": 20, "ma2_max": 100,
    "rsi_length_range": (8, 18),
    "rsi_min_range": (55.0, 65.0),
    "rsi_max_range": (35.0, 45.0),
    "adx_length_range": (8, 18),
    "adx_thr_range": (15.0, 25.0),
    "lookback_range": (2, 10),
    "valwin_range": (5, 15),
    "sl_range": (0.5, 2.0),
    "be_range": (1, 10),
    "tp_long_range": (1.0, 4.0),
    "tp_short_range": (1.0, 4.0),
    "mls_range": (1, 3),
    "cool_range": (10, 100),
    "re_range": (1, 4),
    "postre_range": (0, 3),
}

# Features (mismos que usaste)
features = {
    "enable_long_trades": True,
    "enable_short_trades": True,
    "use_rsi_long": "auto",
    "use_rsi_short": "auto",
    "use_adx_filter": "auto",
    "enable_high_condition": True,
    "enable_low_condition": True,
    "use_validation_window": True,
}

# ============================================================
# PRUEBA 1: MODO SERIE (3 corridas, 1000 trials)
# ============================================================

print("\n" + "="*70)
print("  PRUEBA 1: MODO SERIE (Converger)")
print("="*70)

resultados_serie = []

for corrida in range(3):
    print(f"\n--- Corrida Serie {corrida+1}/3 ---")
    
    optimizador = OptimizadorAutomatico(
        df=df,
        config_base=config_base,
        features=features,
        symbol="SPY",
        timeframe="1d",
        verbose=False
    )
    
    # Forzar modo serie
    optimizador.config_fases["fase_1"]["modo"] = "serie"
    optimizador.config_fases["fase_2"]["modo"] = "serie"
    optimizador.config_fases["fase_3"]["modo"] = "serie"
    optimizador.config_fases["fase_1"]["trials"] = 1000
    optimizador.config_fases["fase_2"]["trials"] = 800
    optimizador.config_fases["fase_3"]["trials_por_corrida"] = 500
    
    # Desactivar convergencia para pruebas consistentes
    optimizador.config_convergencia["activar"] = False
    
    # Ejecutar solo fase 1 (para ser más rápido)
    params = optimizador._fase_exploracion_rapida()
    
    resultados_serie.append({
        "corrida": corrida + 1,
        "best_score": optimizador.historial[-1]["best_score"],
        "rsi_long": params.get("usar_rsi_long", "N/A"),
        "rsi_short": params.get("usar_rsi_short", "N/A"),
        "reentry": params.get("usar_reentry", "N/A"),
        "ma1_length": params.get("ma1_length", "N/A"),
        "ma2_length": params.get("ma2_length", "N/A"),
    })
    
    print(f"   Score: {resultados_serie[-1]['best_score']:.4f}")
    print(f"   RSI Long: {resultados_serie[-1]['rsi_long']}")
    print(f"   RSI Short: {resultados_serie[-1]['rsi_short']}")

# ============================================================
# PRUEBA 2: MODO PARALELO (3 corridas, 1000 trials)
# ============================================================

print("\n" + "="*70)
print("  PRUEBA 2: MODO PARALELO")
print("="*70)

resultados_paralelo = []

for corrida in range(3):
    print(f"\n--- Corrida Paralelo {corrida+1}/3 ---")
    
    optimizador = OptimizadorAutomatico(
        df=df,
        config_base=config_base,
        features=features,
        symbol="SPY",
        timeframe="1d",
        verbose=False
    )
    
    # Forzar modo paralelo
    optimizador.config_fases["fase_1"]["modo"] = "paralelo"
    optimizador.config_fases["fase_2"]["modo"] = "paralelo"
    optimizador.config_fases["fase_3"]["modo"] = "paralelo"
    optimizador.config_fases["fase_1"]["trials"] = 1000
    optimizador.config_fases["fase_2"]["trials"] = 800
    optimizador.config_fases["fase_3"]["trials_por_corrida"] = 500
    
    # Desactivar convergencia
    optimizador.config_convergencia["activar"] = False
    
    # Ejecutar solo fase 1
    params = optimizador._fase_exploracion_rapida()
    
    resultados_paralelo.append({
        "corrida": corrida + 1,
        "best_score": optimizador.historial[-1]["best_score"],
        "rsi_long": params.get("usar_rsi_long", "N/A"),
        "rsi_short": params.get("usar_rsi_short", "N/A"),
        "reentry": params.get("usar_reentry", "N/A"),
        "ma1_length": params.get("ma1_length", "N/A"),
        "ma2_length": params.get("ma2_length", "N/A"),
    })
    
    print(f"   Score: {resultados_paralelo[-1]['best_score']:.4f}")
    print(f"   RSI Long: {resultados_paralelo[-1]['rsi_long']}")
    print(f"   RSI Short: {resultados_paralelo[-1]['rsi_short']}")

# ============================================================
# RESULTADOS COMPARATIVOS
# ============================================================

print("\n" + "="*70)
print("  📊 RESULTADOS COMPARATIVOS")
print("="*70)

# Calcular estadísticas
scores_serie = [r["best_score"] for r in resultados_serie]
scores_paralelo = [r["best_score"] for r in resultados_paralelo]

print("\n┌─────────────┬──────────────┬──────────────┬──────────────┐")
print("│   Corrida   │  Serie       │  Paralelo    │  Diferencia  │")
print("├─────────────┼──────────────┼──────────────┼──────────────┤")
for i in range(3):
    diff = scores_paralelo[i] - scores_serie[i]
    print(f"│     {i+1}     │    {scores_serie[i]:.4f}    │    {scores_paralelo[i]:.4f}    │    {diff:+.4f}    │")
print("├─────────────┼──────────────┼──────────────┼──────────────┤")
print(f"│   Promedio  │   {np.mean(scores_serie):.4f}   │   {np.mean(scores_paralelo):.4f}   │   {np.mean(scores_paralelo)-np.mean(scores_serie):+.4f}   │")
print(f"│   Desv. Std │   {np.std(scores_serie):.4f}   │   {np.std(scores_paralelo):.4f}   │              │")
print("└─────────────┴──────────────┴──────────────┴──────────────┘")

print("\n📈 INTERPRETACIÓN:")
print(f"   • Serie:     Scores entre {min(scores_serie):.4f} y {max(scores_serie):.4f} (dispersión: {np.std(scores_serie):.4f})")
print(f"   • Paralelo:  Scores entre {min(scores_paralelo):.4f} y {max(scores_paralelo):.4f} (dispersión: {np.std(scores_paralelo):.4f})")

if np.std(scores_paralelo) < np.std(scores_serie):
    print("\n   ✅ El modo PARALELO muestra MENOS dispersión (más consistente).")
else:
    print("\n   ℹ️  El modo SERIE muestra MENOS dispersión en esta prueba.")

print("\n" + "="*70)