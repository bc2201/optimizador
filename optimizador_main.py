"""
OPTIMIZADOR DE ESTRATEGIAS - MÓDULO PRINCIPAL
=============================================
Autor: Trading System
Versión: 1.8

Este módulo contiene la lógica central del optimizador:
- Descarga de datos de Binance (cripto) y Yahoo Finance (acciones)
- Cálculo de indicadores técnicos (RSI, ADX, MAs)
- Backtest de estrategia de cruce de medias móviles
- Optimización con Optuna

Dependencias: ccxt, pandas, numpy, optuna, yfinance, plotly
"""

# ============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN GLOBAL
# ============================================================================

import ccxt
import pandas as pd
import numpy as np
import optuna
import threading
import json
import os
from datetime import datetime
from queue import Queue
from data_providers import CCXTProvider, YFinanceProvider


# Módulos propios
from config import CONSTANTS, FEATURES, TIMEFRAME_CONFIG, DEFAULT_RANGES
from indicators import ema, sma, wma, hma, dema, ma, rma, rsi_tv, adx_tv
from backtest import run_backtest, calcular_drawdown_maximo, calcular_score
from reporting import build_ascii_table, generar_reporte_ascii, guardar_reporte_txt, loguear_reporte_en_console, generar_tabla_overfitting
from mis_graficos import generar_grafico, generar_preview_velas



# Cola para GUI
progress_queue = Queue()

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "reportes")
try:
    os.makedirs(output_dir, exist_ok=True)
except Exception:
    output_dir = os.path.join(os.path.expanduser("~"), "Documents", "OptunaTrades")
    os.makedirs(output_dir, exist_ok=True)




# ============================================================================
# CONFIGURACIÓN DE LOGGING Y WARNINGS
# ============================================================================
import warnings
import optuna
import logging

# Silenciar SOLO los warnings de rangos decimales (no los de progreso)
warnings.filterwarnings("ignore", message="The distribution is specified by")

# Configurar Optuna para mostrar progreso pero no warnings internos
optuna.logging.set_verbosity(optuna.logging.INFO)

# Opcional: también silenciar warnings de UserWarning de Optuna
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")




# ==================================
# SECCIÓN 1: Función para guardar el SEED
# ==================================


def guardar_seed(values, config, best_params, metrics, output_files, symbol, timeframe, timestamp):
    seed = {
        "version": "16",
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,

        "gui_values": values,
        "config_ranges": config,

        "optuna_settings": {
            "trials": values.get("trials"),
            "multi_run": values.get("multi_run"),
            "runs": values.get("multi_runs_count"),
            "mode": values.get("modo_opt")
        },

        "random_seed": values.get("random_seed", None),

        "best_params": best_params,
        "metrics": metrics,
        "output_files": output_files
    }

    clean_symbol = symbol.replace("/", "_").replace(":", "_")
    ruta_seed = os.path.join(output_dir, f"{timestamp} Seed {clean_symbol}_{timeframe}.json")

    with open(ruta_seed, "w", encoding="utf-8") as f:
        json.dump(seed, f, indent=4)

    print(f"\n[ÉXITO] Seed generado:\n  {ruta_seed}")



# ============================================================================
# SECCIÓN 2: CONFIGURACIÓN CENTRALIZADA (FEATURES Y CONSTANTES)
# ============================================================================
# FEATURES: Configuración por defecto de los filtros de la estrategia
# CONSTANTS: Parámetros fijos del backtest (comisión, capital, riesgo)
# ============================================================================

# FEATURES movido a config.py
# CONSTANTS movido a config.py



# ============================================================================
# SECCIÓN 3: CONFIGURACIÓN DE TIMEFRAMES
# ============================================================================

#Modulo (TIMEFRAME_CONFIG) movido a config en v19


# ============================================================================
# SECCIÓN 4: CACHÉ DE DATOS
# ============================================================================

_cached_df = None          # DataFrame cacheado
_cached_params = None      # Parámetros que generaron el caché (symbol, timeframe, total_candles, data_source)


# ============================================================================
# SECCIÓN 5: LIMPIEZA DE OUTLIERS PARA YAHOO FINANCE
# ============================================================================
# Yahoo Finance a veces devuelve velas con precios imposibles
# Esta función detecta y elimina esas velas anómalas.
# ============================================================================


def clean_yfinance_outliers(df, symbol, timeframe):
    """
    Limpia outliers de datos de Yahoo Finance para timeframes intradiarios.
    
    Métodos de detección:
        1. Desviación de la mediana móvil (MAD - más robusto que desviación estándar)
        2. Cambio porcentual extremo (ajustado por timeframe)
        3. Consistencia OHLC (high >= open/close >= low)
        4. Detección de "islas" (picos aislados rodeados de precios normales)
        5. Gaps temporales anómalos (velas fuera de horario)
    
    Args:
        df: DataFrame con columnas open, high, low, close, volume
        symbol: Símbolo del activo (para logs)
        timeframe: Timeframe (ej: "5m", "1h")
    
    Returns:
        DataFrame limpio (sin velas outlier)
    """
    if df is None or len(df) < 50:
        return df
    
    # Solo aplicar a timeframes intradiarios (donde Yahoo Finance tiene problemas)
    if not TIMEFRAME_CONFIG.get(timeframe, {}).get("intraday", False):
        return df
    
    df = df.copy()
    original_len = len(df)
    
    # --- MÉTODO 1: Mediana móvil (MAD) ---
    window_size = max(21, min(51, len(df) // 20))  # Ventana adaptativa al tamaño de datos
    
    rolling_median = df['close'].rolling(window=window_size, min_periods=window_size//2, center=False).median()
    rolling_median = rolling_median.bfill().ffill()  # Corregido: deprecated method
    
    mad = (df['close'] - rolling_median).abs().rolling(window=window_size, min_periods=window_size//2).median()
    mad = mad.bfill().ffill().clip(lower=0.01)  # Corregido: deprecated method
    
    z_mad = (df['close'] - rolling_median).abs() / mad
    outlier_mask = z_mad > 4.0  # Umbral: 4 desviaciones MAD
    
    # --- MÉTODO 2: Cambio porcentual extremo ---
    pct_threshold = TIMEFRAME_CONFIG.get(timeframe, {}).get("pct_threshold", 0.08)
    pct_change = df['close'].pct_change().abs()
    outlier_mask |= (pct_change > pct_threshold)
    
    # Verificar rango high/low
    high_low_range = (df['high'] - df['low']) / df['low'].clip(lower=0.01)
    outlier_mask |= (high_low_range > pct_threshold * 1.5)
    
    # --- MÉTODO 3: Consistencia OHLC ---
    invalid_ohlc = (df['high'] < df['low']) | (df['high'] <= 0) | (df['low'] <= 0)
    invalid_ohlc |= (df['open'] < df['low']) | (df['open'] > df['high'])
    invalid_ohlc |= (df['close'] < df['low']) | (df['close'] > df['high'])
    outlier_mask |= invalid_ohlc
    
    # --- MÉTODO 4: Detección de "islas" (picos aislados) ---
    # Una "isla" es una vela cuyo precio es significativamente diferente
    # al de las 10 velas anteriores Y las 10 siguientes
    for i in range(5, len(df) - 5):
        if outlier_mask.iloc[i]:
            continue
            
        prev_median = df['close'].iloc[max(0, i-10):i].median()
        next_median = df['close'].iloc[i+1:min(len(df), i+11)].median()
        current = df['close'].iloc[i]
        
        dev_from_prev = abs(current - prev_median) / max(prev_median, 0.01)
        dev_from_next = abs(current - next_median) / max(next_median, 0.01)
        
        if dev_from_prev > 0.08 and dev_from_next > 0.08:
            side_diff = abs(prev_median - next_median) / max(prev_median, next_median, 0.01)
            if side_diff < 0.03:  # Los lados son similares, el punto medio es el outlier
                outlier_mask.iloc[i] = True
    
    # --- MÉTODO 5: Gaps temporales anómalos ---
    if len(df) > 1:
        time_diff = df.index.to_series().diff().dt.total_seconds() / 60
        expected_interval = TIMEFRAME_CONFIG.get(timeframe, {}).get("interval_min", 5)
        
        large_gap = time_diff > (expected_interval * 2)
        
        if large_gap.any():
            gap_indices = df.index[large_gap.fillna(False)]
            for gap_idx in gap_indices:
                if gap_idx in df.index:
                    pos = df.index.get_loc(gap_idx)
                    if pos < len(df) - 1:
                        next_close = df['close'].iloc[pos + 1]
                        prev_close = df['close'].iloc[pos]
                        if abs(next_close - prev_close) / max(prev_close, 0.01) > pct_threshold:
                            outlier_mask.iloc[pos + 1] = True
    
    outlier_mask = outlier_mask.fillna(False)
    
    # --- Aplicar limpieza con límite de seguridad (máx 15% de velas) ---
    n_outliers = outlier_mask.sum()
    max_outliers = int(len(df) * 0.15)
    
    if n_outliers > max_outliers:
        print(f"[LIMPIEZA] ADVERTENCIA: {n_outliers} outliers detectados (>15%). Aplicando solo los más extremos.")
        severity = z_mad.fillna(0)
        outlier_mask = outlier_mask & (severity > severity.quantile(0.95))
        n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        print(f"\n[LIMPIEZA] {symbol} {timeframe}: Eliminando {n_outliers} velas anómalas "
              f"({n_outliers/len(df)*100:.1f}% del total)")
        
        outlier_indices = df.index[outlier_mask]
        for idx in list(outlier_indices)[:5]:
            print(f"  → Outlier en {idx}: close={df.loc[idx, 'close']:.2f}")
        
        df = df.loc[~outlier_mask]
        print(f"[LIMPIEZA] Resultado: {len(df)} velas (eliminadas {original_len - len(df)})")
    else:
        print(f"[LIMPIEZA] No se detectaron outliers en {symbol}")
    
    return df.sort_index()



# ============================================================================
# SECCIÓN 6: OBTENCIÓN DE DATOS (CON CACHÉ)
# ============================================================================
# Capa unificada que maneja:
#   - Cripto → Binance vía fetch_binance_ohlcv()
#   - Acciones → Yahoo Finance vía YFinanceProvider + limpieza de outliers
#   - Caché automático para evitar descargas repetidas
# ============================================================================


def get_data_efficiently(symbol, timeframe, total_candles, data_source):
    """
    Obtiene datos OHLCV de forma eficiente con caché.
    
    Args:
        symbol: Símbolo del activo (ej: "BTC/USDT" o "AAPL")
        timeframe: Timeframe (ej: "5m", "1h", "1d")
        total_candles: Cantidad de velas a obtener
        data_source: Fuente de datos ("Cripto (Binance)" o "Acciones (Yahoo Finance)")
    
    Returns:
        DataFrame con columnas open, high, low, close, volume
    """
    global _cached_df, _cached_params

    current_params = (symbol, timeframe, total_candles, data_source)


    # Retornar caché si coincide
    # -----------------------------
    if _cached_df is not None and _cached_params == current_params:
        print(f"\n[CACHE] Utilizando {len(_cached_df)} velas ya cargadas.")
        return _cached_df

    print(f"\n[API] Descargando datos nuevos ({data_source})...")


    # --- CRIPTO: Binance ---
    # -----------------------------

    if data_source.startswith("Cripto"):
        df = fetch_binance_ohlcv(symbol, timeframe, total_candles)


    # --- ACCIONES: Yahoo Finance ---
    # -----------------------------

    elif data_source.startswith("Acciones"):
        # Advertencia para timeframes intradiarios (datos pueden ser corruptos)
        if timeframe != "1d":
            print(f"\n[ADVERTENCIA] Acciones con timeframe {timeframe} - Los datos de Yahoo Finance pueden ser corruptos.")
            print(f"            Se recomienda usar timeframe 1D o cambiar a Cripto (Binance).")

        provider = YFinanceProvider()
        df = provider.get_ohlc(symbol, timeframe, total_candles)

        if df is not None and len(df) > 0:
            print(f"[LIMPIEZA] Antes: {len(df)} velas")
            df = clean_yfinance_outliers(df, symbol, timeframe)
            if df is not None and len(df) > 0:
                print(f"[LIMPIEZA] Después: {len(df)} velas")
            else:
                print("[LIMPIEZA] ADVERTENCIA: La limpieza eliminó TODAS las velas.")


    else:
        raise ValueError(f"Fuente de datos no soportada: {data_source}")



    # Validación mínima
    # -----------------------------

    if df is None or len(df) < 50:
        print(f"\n[ERROR] No se pudieron obtener suficientes velas. Obtenidas: {len(df) if df is not None else 0}\n")
        return None

    # Recortar al número exacto solicitado
    # -----------------------------

    if len(df) > total_candles:
        df = df.tail(total_candles)
        print(f"[INFO] Recortado a las últimas {total_candles} velas")

    # Actualizar caché
    # -----------------------------

    _cached_df = df
    _cached_params = current_params

    print(f"\n[ÉXITO] Descargadas y limpiadas {len(df)} velas para {symbol} {timeframe}")
    print(f"[RANGO] {df.index[0]} → {df.index[-1]}")

    return df



# ============================================================================
# SECCIÓN 7: DESCARGA DE DATOS BINANCE (PRIORIDAD RECENCIA)
# ============================================================================
# Este método asegura obtener las últimas N velas, incluso si hay huecos en los datos.
# ============================================================================


def fetch_binance_ohlcv(symbol="SOL/USDT", timeframe="5m", total_candles=10000):
    """
    Descarga velas de Binance asegurando obtener las más recientes.
    
    Estrategia:
        1. Retrocede 50% más del total solicitado para cubrir huecos
        2. Descarga en bloques de 1000 velas hasta llegar al presente
        3. Recorta a las últimas 'total_candles' velas
    
    Args:
        symbol: Par de trading (ej: "BTC/USDT")
        timeframe: Timeframe (ej: "1m", "5m", "1h")
        total_candles: Cantidad de velas a obtener
    
    Returns:
        DataFrame con datos OHLCV
    """
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    
    duration_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - (total_candles * duration_ms * 1.5)
    
    all_ohlcv = []
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(since), limit=1000)
        if not ohlcv:
            break
        
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        
        # Detener cuando alcanzamos el presente
        if ohlcv[-1][0] >= (exchange.milliseconds() - duration_ms * 2):
            break

    # Quedarse solo con las últimas N velas
    all_ohlcv = all_ohlcv[-total_candles:]

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)



# ============================================================================
# SECCIÓN 8: VERIFICACIÓN DE DISPONIBILIDAD DE VELAS
# ============================================================================


def check_available_candles(symbol, timeframe, requested, data_source):
    """
    Verifica cuántas velas están disponibles sin descargar todo el histórico.
    
    Returns:
        Dict con: exchange, symbol, name, last_price, available, start, end
    """
    symbol = symbol.upper().strip()
    result = {
        "exchange": "Binance" if data_source.startswith("Cripto") else "N/A",
        "symbol": symbol,
        "name": symbol,
        "last_price": 0.0,
        "available": 0,
        "start": None,
        "end": None
    }

    
    # --- CRIPTO: Binance ---
    # -----------------------------

    if data_source.startswith("Cripto"):
        try:
            exchange = ccxt.binance({"enableRateLimit": True})
            duration_ms = exchange.parse_timeframe(timeframe) * 1000
            since = exchange.milliseconds() - (requested * duration_ms * 1.5)

            all_ohlcv = []
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(since), limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if ohlcv[-1][0] >= (exchange.milliseconds() - duration_ms * 2):
                    break

            if all_ohlcv:
                final_data = all_ohlcv[-requested:]
                df = pd.DataFrame(final_data, columns=["ts", "o", "h", "l", "c", "v"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                
                try:
                    markets = exchange.load_markets()
                    if symbol in markets:
                        result["name"] = markets[symbol].get('info', {}).get('baseAsset', symbol)
                except:
                    pass

                result["available"] = len(df)
                result["start"] = df["ts"].iloc[0]
                result["end"] = df["ts"].iloc[-1]
                result["last_price"] = float(df["c"].iloc[-1])
        except Exception as e:
            print(f"[ERROR check_candles Cripto]: {e}")

    
    # --- ACCIONES: Yahoo Finance ---
    # -----------------------------

    elif data_source.startswith("Acciones"):
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(symbol)
            info = ticker_obj.info
            result["name"] = info.get("longName", symbol)
            result["exchange"] = info.get("exchange", "Yahoo Finance")
            result["last_price"] = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0.0

            from data_providers import YFinanceProvider
            provider = YFinanceProvider()
            df = provider.get_ohlc(symbol, timeframe, requested)

            if df is not None and not df.empty:
                df = df.dropna()
                result["available"] = len(df)
                result["start"] = df.index[0]
                result["end"] = df.index[-1]
                if result["last_price"] == 0.0 and "close" in df.columns:
                    result["last_price"] = float(df["close"].iloc[-1])
        except Exception as e:
            print(f"[ERROR check_candles Acciones]: {e}")

    return result



# ============================================================================
# SECCIÓN 9: INDICADORES TÉCNICOS
# ============================================================================
# Implementaciones de medias móviles, RSI, ADX
# ============================================================================

#Modulos "ema, sma, wma, hma, dema, ma, rma, rsi_tv, adx_tv" movidos a indicators.py


# ============================================================================
# SECCIÓN 10: BACKTEST DE LA ESTRATEGIA
# ============================================================================
# Estrategia de cruce de dos medias móviles con múltiples filtros:
#   - RSI (sobrecompra/sobreventa)
#   - ADX (fuerza de tendencia)
#   - High/Lookback (nuevos máximos/mínimos)
#   - HTF Filter (tendencia de timeframe superior)
#   - Stop Loss / Take Profit / Break Even
#   - Cooldown (enfriamiento tras racha de pérdidas)
#   - Reentradas y post-crossover entries
# ============================================================================


# def _get_indicator movido a backtest (engine.py)
# Def run_backtest movido a backtest (engine.py)


# ============================================================================
# SECCIÓN 11: FUNCIONES DE MÉTRICAS Y SCORE
# ============================================================================

# Def calcular_drawdown_maximo y def calcular_score movidos a backtest.py



# ============================================================================
# SECCIÓN 12: OPTUNA - FUNCIÓN OBJETIVO
# ============================================================================
# Optuna explora el espacio de hiperparámetros buscando maximizar el score.
# Cada trial sugiere valores dentro de los rangos configurados.
# ============================================================================

def objective(trial, df, config, features, metrics_config, study_cache, study_lock):
    """Función objetivo de Optuna: ejecuta un backtest y retorna el score."""
    
    params = {}

    def resolve(key, trial_key):
        """Resuelve si un feature está activo (True/False o Auto)."""
        mode = features.get(key, False)
        if mode == "auto":
            return trial.suggest_categorical(trial_key, [True, False])
        return bool(mode)


    # --- MEDIAS MÓVILES ---
    # ------------------------

    params["ma1_type"] = trial.suggest_categorical("ma1_type", config["tipos_ma"])
    params["ma2_type"] = trial.suggest_categorical("ma2_type", config["tipos_ma"])
    params["ma1_length"] = trial.suggest_int("ma1_length", config["ma1_min"], config["ma1_max"])
    params["ma2_length"] = trial.suggest_int("ma2_length", config["ma2_min"], config["ma2_max"])

    # Validación: MA1 debe ser más rápida que MA2
    if params["ma1_length"] >= params["ma2_length"]:
        return 0.0


    # --- RSI ---
    # ------------------------
    usar_rsi_long = resolve("use_rsi_long", "usar_rsi_long")
    usar_rsi_short = resolve("use_rsi_short", "usar_rsi_short")

    # RSI Length (solo si al menos uno está activo)
    if usar_rsi_long or usar_rsi_short:
        params["rsi_length"] = trial.suggest_int("rsi_length", *config["rsi_length_range"])
    else:
        params["rsi_length"] = 14

    # RSI min (solo si LONG está activo)
    if usar_rsi_long:
        params["rsi_min"] = trial.suggest_float("rsi_min", *config["rsi_min_range"], step=0.1)
        # Validación: rsi_min debe ser >50 (sobrecompra)
        if params["rsi_min"] <= 50:
            return 0.0
    else:
        params["rsi_min"] = 55.0  # valor por defecto (no se usará)

    # RSI max (solo si SHORT está activo)
    if usar_rsi_short:
        params["rsi_max"] = trial.suggest_float("rsi_max", *config["rsi_max_range"], step=0.1)
        # Validación: rsi_max debe ser <50 (sobreventa)
        if params["rsi_max"] >= 50:
            return 0.0
    else:
        params["rsi_max"] = 45.0  # valor por defecto (no se usará)


    # --- ADX ---
    # ------------------------

    usar_adx = resolve("use_adx_filter", "usar_adx")
    if usar_adx:
        params["adx_length"] = trial.suggest_int("adx_length", *config["adx_length_range"])
        params["adx_threshold"] = trial.suggest_float("adx_threshold", *config["adx_thr_range"], step=0.1)
    else:
        params["adx_length"] = 14
        params["adx_threshold"] = 18.0


    # --- HIGH/LOW ---
    # ------------------------

    usar_high = resolve("enable_high_condition", "usar_high")
    usar_low = resolve("enable_low_condition", "usar_low")
    if usar_high or usar_low:
        params["lookback"] = trial.suggest_int("lookback", *config["lookback_range"])
    else:
        params["lookback"] = 5


    # --- VALIDATION WINDOW ---
    # ------------------------

    if features["use_validation_window"]:
        params["validation_window"] = trial.suggest_int("validation_window", *config["valwin_range"])
    else:
        params["validation_window"] = 20



    # --- HTF FILTER ---
    # ------------------------

    usar_htf = resolve("use_htf_filter", "usar_htf")
    if usar_htf:
        params["htf_length"] = trial.suggest_int("htf_length", *config["htf_length_range"])
    else:
        params["htf_length"] = 30


    # --- STOP LOSS ---
    # ------------------------

    usar_sl = resolve("use_stop_loss", "usar_sl")
    if usar_sl:
        params["stop_loss_pct"] = trial.suggest_float("stop_loss_pct", *config["sl_range"], step=0.1)
    else:
        params["stop_loss_pct"] = 1.0


    # --- BREAK EVEN ---
    # ------------------------

    usar_be = resolve("activar_stop_be", "usar_be")
    if usar_be:
        params["velas_para_be"] = trial.suggest_int("velas_para_be", *config["be_range"])
    else:
        params["velas_para_be"] = 3


    # --- TAKE PROFIT ---
    # ------------------------

    usar_tp_long = resolve("use_take_profit_long", "usar_tp_long")
    usar_tp_short = resolve("use_take_profit_short", "usar_tp_short")
    if usar_tp_long:
        params["tp_long_pct"] = trial.suggest_float("tp_long_pct", *config["tp_long_range"], step=0.1)
    else:
        params["tp_long_pct"] = 2.0
    if usar_tp_short:
        params["tp_short_pct"] = trial.suggest_float("tp_short_pct", *config["tp_short_range"], step=0.1)
    else:
        params["tp_short_pct"] = 2.0


    # --- COOLDOWN ---
    # ------------------------

    usar_cooldown = resolve("enable_cooldown", "usar_cooldown")
    if usar_cooldown:
        params["max_losing_streak"] = trial.suggest_int("max_losing_streak", *config["mls_range"])
        params["cooldown_bars"] = trial.suggest_int("cooldown_bars", *config["cool_range"])
    else:
        params["max_losing_streak"] = 1
        params["cooldown_bars"] = 50


    # --- REENTRADAS ---
    # ------------------------

    usar_reentry = resolve("enable_reentry", "usar_reentry")
    usar_post_re = resolve("enable_post_crossover_entry", "usar_post_re")
    if usar_reentry:
        params["max_reentries_allowed"] = trial.suggest_int("max_reentries_allowed", *config["re_range"])
    else:
        params["max_reentries_allowed"] = 0
    if usar_post_re:
        params["max_post_reentries"] = trial.suggest_int("max_post_reentries", *config["postre_range"])
    else:
        params["max_post_reentries"] = 0


    # --- EJECUTAR BACKTEST ---
    # ------------------------
    profit_factor, equity_curve, trades = run_backtest(
        df,
        **params,
        enable_long_trades=features["enable_long_trades"],
        enable_short_trades=features["enable_short_trades"],
        use_rsi_long=usar_rsi_long,
        use_rsi_short=usar_rsi_short,
        use_adx_filter=usar_adx,
        enable_high_condition=usar_high,
        enable_low_condition=usar_low,
        use_validation_window=features["use_validation_window"],
        use_htf_filter=usar_htf,
        use_stop_loss=usar_sl,
        activar_stop_be=usar_be,
        enable_cooldown=usar_cooldown,
        enable_reentry=usar_reentry,
        enable_post_crossover_entry=usar_post_re,
        use_take_profit_long=usar_tp_long,
        use_take_profit_short=usar_tp_short,
        commission_pct=CONSTANTS["commission_pct"],
        initial_capital=CONSTANTS["initial_capital"],
        risk_pct_per_trade=CONSTANTS["risk_pct_per_trade"],
        htf_tf=CONSTANTS["htf_tf"],
        htf_type=CONSTANTS["htf_type"],
        _cache=study_cache,
        _lock=study_lock,
    )

    # Corte duro por cantidad mínima de trades
    min_trades = int(metrics_config.get("min_trades", 30))
    if len(trades) < min_trades:
        return 0.0

    return calcular_score(profit_factor, trades, equity_curve, metrics_config)



# ============================================================================
# SECCIÓN 13: MOTOR DE OPTIMIZACIÓN (SINGLE RUN)
# ============================================================================

def run_single_optuna(df, config, n_trials, modo_paralelo, features, stop_event=None, metrics_config=None):
    """
    Ejecuta una instancia de optimización con Optuna.
    
    Args:
        df: DataFrame con datos históricos
        config: Configuración de rangos desde GUI
        n_trials: Cantidad de trials a ejecutar
        modo_paralelo: True = paralelo (n_jobs=-1), False = serie
        features: Diccionario de features (con valores True/False/"auto")
        stop_event: Evento para cancelación desde GUI
        metrics_config: Configuración de métricas para el score
    
    Returns:
        Tuple (best_score, best_params)
    """
    if metrics_config is None:
        metrics_config = {
            "use_pf": True, "peso_pf": 50.0,
            "use_winrate": True, "peso_winrate": 30.0,
            "use_drawdown": True, "peso_drawdown": 20.0
        }

    study_cache = {}
    study_lock = threading.Lock()

    def actualizar_barra_callback(study, trial):
        progreso = min((trial.number + 1) / n_trials, 1.0)
        progress_queue.put(progreso)
        if stop_event is not None and stop_event.is_set():
            study.stop()

    sampler = optuna.samplers.TPESampler(seed=None)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, df, config, features, metrics_config, study_cache, study_lock),
        n_trials=n_trials,
        n_jobs=-1 if modo_paralelo else 1,
        callbacks=[actualizar_barra_callback]
    )

    if not study.trials:
        return 0.0, {}
    return study.best_value, study.best_params



# ============================================================================
# SECCIÓN 14: FUNCIONES DE SOPORTE PARA GUI
# ============================================================================
# Conversión de rangos, validaciones, generación de reportes
# ============================================================================


def get_range(values, key_min, key_max, default_min, default_max, cast_type=float):
    """Extrae un rango (min, max) de los valores del GUI."""
    raw_min = values.get(key_min, "")
    raw_max = values.get(key_max, "")
    try:
        vmin = cast_type(raw_min)
        vmax = cast_type(raw_max)
        if vmin >= vmax:
            return default_min, default_max
        return vmin, vmax
    except:
        return default_min, default_max


def build_config(values):
    """Construye el diccionario de configuración de rangos desde el GUI."""
    tipos_ma = []
    if values.get("ma_ema"):
        tipos_ma.append("EMA")
    if values.get("ma_sma"):
        tipos_ma.append("SMA")
    if values.get("ma_wma"):
        tipos_ma.append("WMA")
    if values.get("ma_hma"):
        tipos_ma.append("HMA")
    if values.get("ma_dema"):
        tipos_ma.append("DEMA")

    if not tipos_ma:
        return None

    return {
        "tipos_ma": tipos_ma,
        "ma1_min": int(values["ma1_min"]),
        "ma1_max": int(values["ma1_max"]),
        "ma2_min": int(values["ma2_min"]),
        "ma2_max": int(values["ma2_max"]),
        "rsi_length_range": get_range(values, "rsi_length_min", "rsi_length_max", 2, 50, int),
        "rsi_min_range": get_range(values, "rsi_min_min", "rsi_min_max", 50.0, 80.0, float),
        "rsi_max_range": get_range(values, "rsi_max_min", "rsi_max_max", 5.0, 50.0, float),
        "adx_length_range": get_range(values, "adx_length_min", "adx_length_max", 2, 50, int),
        "adx_thr_range": get_range(values, "adx_threshold_min", "adx_threshold_max", 5.0, 70.0, float),
        "lookback_range": get_range(values, "lookback_min", "lookback_max", 1, 15, int),
        "valwin_range": get_range(values, "validation_window_min", "validation_window_max", 1, 15, int),
        "htf_length_range": get_range(values, "htf_length_min", "htf_length_max", 10, 60, int),
        "sl_range": get_range(values, "stop_loss_min", "stop_loss_max", 0.3, 10.0, float),
        "be_range": get_range(values, "velas_para_be_min", "velas_para_be_max", 1, 10, int),
        "tp_long_range": get_range(values, "tp_long_min", "tp_long_max", 0.3, 99.0, float),
        "tp_short_range": get_range(values, "tp_short_min", "tp_short_max", 0.3, 99.0, float),
        "mls_range": get_range(values, "max_losing_streak_min", "max_losing_streak_max", 1, 4, int),
        "cool_range": get_range(values, "cooldown_bars_min", "cooldown_bars_max", 10, 300, int),
        "re_range": get_range(values, "max_reentries_min", "max_reentries_max", 1, 4, int),
        "postre_range": get_range(values, "max_post_reentries_min", "max_post_reentries_max", 1, 4, int),
    }


def log_selected_parameters(values):
    print("\n==============================")
    print(" PARÁMETROS SETEADOS (GUI)")
    print("==============================\n")

    # --- FEATURES ---
    features = [
        ("RSI Long", "use_rsi_long", ["rsi_length_min","rsi_length_max","rsi_min_min","rsi_min_max"]),
        ("RSI Short", "use_rsi_short", ["rsi_length_min","rsi_length_max","rsi_max_min","rsi_max_max"]),
        ("ADX Filter", "use_adx_filter", ["adx_length_min","adx_length_max","adx_threshold_min","adx_threshold_max"]),
        ("High Condition", "enable_high_condition", ["lookback_min","lookback_max"]),
        ("Low Condition", "enable_low_condition", ["lookback_min","lookback_max"]),
        ("Validation Window", "use_validation_window", ["validation_window_min","validation_window_max"]),
        ("HTF Filter", "use_htf_filter", ["htf_length_min","htf_length_max"]),
        ("Stop Loss", "use_stop_loss", ["stop_loss_min","stop_loss_max"]),
        ("Break Even", "activar_stop_be", ["velas_para_be_min","velas_para_be_max"]),
        ("Take Profit Long", "use_take_profit_long", ["tp_long_min","tp_long_max"]),
        ("Take Profit Short", "use_take_profit_short", ["tp_short_min","tp_short_max"]),
        ("Cooldown", "enable_cooldown", ["max_losing_streak_min","max_losing_streak_max","cooldown_bars_min","cooldown_bars_max"]),
        ("Reentry", "enable_reentry", ["max_reentries_min","max_reentries_max"]),
        ("Post Crossover Entry", "enable_post_crossover_entry", ["max_post_reentries_min","max_post_reentries_max"]),
    ]

    for label, key, params in features:
        if values.get(key):
            print(f"✔ {label}")
            for p in params:
                if p in values and values[p] != "":
                    print(f"    {p}: {values[p]}")
            print("")

    # --- MOVING AVERAGES ---
    print("✔ Tipos de medias móviles:")
    ma_types = ["ma_ema","ma_sma","ma_wma","ma_hma","ma_dema"]
    for ma in ma_types:
        if values.get(ma):
            print(f"    {ma.replace('ma_','').upper()}")

    print(f"\nMA1 rango: {values.get('ma1_min')} → {values.get('ma1_max')}")
    print(f"MA2 rango: {values.get('ma2_min')} → {values.get('ma2_max')}")

    print("\n==============================\n")




# ============================================================================
# SECCIÓN 15: FUNCIONES DE REPORTE (ASCII)
# ============================================================================


def build_ascii_table(rows, title=None):
    """Construye una tabla ASCII para el reporte."""
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    col_widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(row):
        cells = []
        for i, cell in enumerate(row):
            cells.append(str(cell).ljust(col_widths[i]))
        return "| " + " | ".join(cells) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    lines = []
    if title:
        title_line = f"| {title}".ljust(len(sep) - 1) + "|"
        lines.append(title_line)
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)



# ============================================================================
# SECCIÓN 16: GENERACIÓN DE REPORTE ASCII COMPLETO
# ============================================================================


# Movido a ascii_reports.py:

# def _bool_mode
# def _range_str
# def generar_reporte_ascii
# def guardar_reporte_txt
# def loguear_reporte_en_console



# ============================================================================
# SECCIÓN 17: FUNCIÓN PRINCIPAL (OPTIMIZACIÓN DESDE GUI)
# ============================================================================


def run_optuna_with_gui(values, stop_event=None):
    print("\n=== INICIANDO PROCESO DE OPTIMIZACIÓN ===\n")

    log_selected_parameters(values) #Log de parametros elegidos


    # 1) Configuración inicial
    config = build_config(values)
    if config is None:
        print("ERROR: Debes seleccionar al menos un tipo de MA.")
        return

    # 2) Construir features local (sin tocar el global FEATURES)
    features = {key: values[key] if key in values else FEATURES[key] for key in FEATURES}

    # Traducir checkboxes Auto del GUI a valor "auto" en features
    AUTO_MAP = {
        "use_rsi_long":                "auto_rsi_long",
        "use_rsi_short":               "auto_rsi_short",
        "use_adx_filter":              "auto_adx_filter",
        "enable_high_condition":       "auto_high_condition",
        "enable_low_condition":        "auto_low_condition",
        "use_htf_filter":              "auto_htf_filter",
        "use_stop_loss":               "auto_stop_loss",
        "activar_stop_be":             "auto_stop_be",
        "use_take_profit_long":        "auto_tp_long",
        "use_take_profit_short":       "auto_tp_short",
        "enable_cooldown":             "auto_cooldown",
        "enable_reentry":              "auto_reentry",
        "enable_post_crossover_entry": "auto_post_crossover",
    }
    for feat_key, auto_key in AUTO_MAP.items():
        if values.get(auto_key, False):
            features[feat_key] = "auto"

    # Loguear features en modo Auto
    autos = [k for k, v in features.items() if v == "auto"]
    if autos:
        print(f"[AUTO] Features en modo automático: {', '.join(autos)}\n")

    # 2b) Construir metrics_config desde los valores del GUI
    def _peso(key, default):
        try: return float(values.get(key, default))
        except: return float(default)

    def _int(key, default):
        try: return int(values.get(key, default))
        except: return int(default)

    metrics_config = {
        "use_pf":        values.get("metric_pf", True),
        "peso_pf":       _peso("peso_pf", 50.0),
        "use_winrate":   values.get("metric_winrate", True),
        "peso_winrate":  _peso("peso_winrate", 30.0),
        "use_drawdown":  values.get("metric_drawdown", True),
        "peso_drawdown": _peso("peso_drawdown", 20.0),
        "use_n_trades":  values.get("metric_n_trades", False),
        "peso_n_trades": _peso("peso_n_trades", 0.0),
        "min_trades":    _int("min_trades", 30),
    }

    # 3) Variables de control y Overfitting
    symbol = values["symbol"].upper().strip()
    timeframe = values["timeframe"]
    total_candles = int(values["candles"])
    n_trials = int(values["trials"])
    usar_multi = values["multi_run"]
    n_runs = int(values["multi_runs_count"]) if usar_multi else 1
    modo_paralelo = values["modo_paralelo"]
    
    # --- LÓGICA DE OVERFITTING ---
    use_oos = values.get("use_oos_validation", False)
    try:
        train_pct = float(values.get("oos_train_pct", 70)) / 100.0
    except:
        train_pct = 0.70

    # ============================================================
    # 4) Descarga Eficiente (CRIPTO o ACCIONES)
    # ============================================================
    # 4) Descarga Eficiente
    try:
        df_full = get_data_efficiently(symbol, timeframe, total_candles, values["data_source"])

        if df_full is None or len(df_full) < 50:
            print("\n[ERROR] No se pudieron obtener suficientes velas.\n")
            return

        # Limpiar NaN
        df_full = df_full.dropna().copy()

        # ============================
        # Mostrar rango descargado
        # ============================
        inicio = df_full.index[0]
        fin = df_full.index[-1]
        print(f"[INFO] Rango descargado: {inicio} → {fin}")

    except Exception as e:
        print(f"\n[ERROR] al obtener datos ({values['data_source']}): {e}\n")
        return





    # 5) Partición de Datos (Train/Test)
    if use_oos:
        split_idx = int(len(df_full) * train_pct)
        df_train = df_full.iloc[:split_idx]
        df_test = df_full.iloc[split_idx:]
        print(f"[INFO] Modo IS/OOS Activo. Entrenamiento: {len(df_train)} velas | Validación: {len(df_test)} velas")
    else:
        df_train = df_full
        df_test = df_full # Se valida con los mismos datos si no hay split
        print(f"[INFO] Modo Normal. Optimizando sobre {len(df_full)} velas.")


    # 6) Bucle de Corridas
    resultados = []

    for i in range(n_runs):
        if usar_multi: print(f"\n--- CORRIDA {i+1}/{n_runs} ---")

        # Abortar entre corridas si el usuario canceló
        if stop_event is not None and stop_event.is_set():
            print("\n[INFO] Cancelación detectada. Abortando corridas pendientes.\n")
            break

        progress_queue.put(0)

        pf_train, best_params = run_single_optuna(df_train, config, n_trials, modo_paralelo, features, stop_event, metrics_config)

        # Si se canceló o no hubo trials, saltar esta corrida
        if not best_params:
            print("[INFO] Sin resultados válidos en esta corrida.")
            continue

        # ─── TRADUCCIÓN DE LLAVES DE OPTUNA ───
        MAPEO_AUTO_BACKTEST = {
            "usar_rsi_long":  "use_rsi_long",
            "usar_rsi_short": "use_rsi_short",
            "usar_adx":       "use_adx_filter",
            "usar_high":      "enable_high_condition",
            "usar_low":       "enable_low_condition",
            "usar_htf":       "use_htf_filter",
            "usar_sl":        "use_stop_loss",
            "usar_be":        "activar_stop_be",
            "usar_tp_long":   "use_take_profit_long",
            "usar_tp_short":  "use_take_profit_short",
            "usar_cooldown":  "enable_cooldown",
            "usar_reentry":   "enable_reentry",
            "usar_post_re":   "enable_post_crossover_entry"
        }

        cleaned_best_params = {}
        for k, v in best_params.items():
            if k in MAPEO_AUTO_BACKTEST:
                cleaned_best_params[MAPEO_AUTO_BACKTEST[k]] = v
            else:
                cleaned_best_params[k] = v

        # Removemos las opciones que digan "auto"
        cleaned_features = {k: v for k, v in features.items() if v != "auto"}

        # --- NUEVO: Backtest en TRAIN para obtener métricas de entrenamiento ---
        pf_train_backtest, equity_curve_train, trades_train = run_backtest(df_train, **cleaned_best_params, **cleaned_features, **CONSTANTS)

        # Calcular métricas de TRAIN
        winrate_train = len([t for t in trades_train if t['net_pnl'] > 0]) / len(trades_train) * 100 if trades_train else 0
        drawdown_train = calcular_drawdown_maximo(list(equity_curve_train))
        n_trades_train = len(trades_train)

        # --- VALIDACIÓN FINAL sobre TEST ---
        pf_oos, equity_curve_test, trades_test = run_backtest(df_test, **cleaned_best_params, **cleaned_features, **CONSTANTS)

        # Calcular métricas de TEST
        winrate_test = len([t for t in trades_test if t['net_pnl'] > 0]) / len(trades_test) * 100 if trades_test else 0
        drawdown_test = calcular_drawdown_maximo(list(equity_curve_test))
        n_trades_test = len(trades_test)

        resultados.append({
            "pf_train": pf_train_backtest,
            "pf_test": pf_oos,
            "winrate_train": winrate_train,
            "winrate_test": winrate_test,
            "drawdown_train": drawdown_train,
            "drawdown_test": drawdown_test,
            "trades_train": n_trades_train,
            "trades_test": n_trades_test,
            "params": cleaned_best_params,
            "equity": equity_curve_test,
            "trades": trades_test,
            "is_oos": use_oos
        })

    # Guardia: si se canceló antes de completar alguna corrida
    if not resultados:
        print("\n[INFO] No hay resultados para reportar (optimización cancelada antes de completar un ciclo).\n")
        return



    #---------------------------------------------------------------------------------------------------------------------------------------

    # ============================================================
    # ELEGIR LA MEJOR CORRIDA Y BACKTEST FINAL
    # ============================================================
    
    # Elegir la mejor corrida por pf_test (validación)
    mejor_corrida = max(resultados, key=lambda r: r["pf_test"])
    best_params = mejor_corrida["params"]
    
    # Limpiar features (quitar "auto")
    cleaned_features = {k: v for k, v in features.items() if v != "auto"}
    
    # ============================================================
    # BACKTEST FINAL SOBRE DATOS COMPLETOS (df_full)
    # ============================================================
    if use_oos:
        print("\n[INFO] Overfitting activado - Generando resultados finales sobre el total de datos...")
        # Ejecutar backtest con los mejores parámetros sobre TODOS los datos
        pf_final, equity_curve_final, trades_final = run_backtest(
            df_full,  # <-- DATOS COMPLETOS (1000 velas)
            **best_params,
            **cleaned_features,
            **CONSTANTS
        )
        print(f"[INFO] Resultados finales (sobre {len(df_full)} velas):")
        print(f"   Profit Factor: {pf_final:.2f}")
        print(f"   Trades totales: {len(trades_final)}")
    else:
        print("\n[INFO] Modo normal - Usando resultados de la mejor corrida...")
        # Comportamiento original: usar los trades que ya vienen de la optimización
        pf_final = mejor_corrida["pf_test"]
        equity_curve_final = mejor_corrida["equity"]
        trades_final = mejor_corrida["trades"]
        print(f"[INFO] Resultados (sobre {len(df_full)} velas):")
        print(f"   Profit Factor: {pf_final:.2f}")
        print(f"   Trades totales: {len(trades_final)}")
    
    # ============================================================
    # INFO DE ESPACIO DE BÚSQUEDA (opcional)
    # ============================================================
    search_space_info = {
        "dim_totales": "N/A",
        "complejidad": "N/A",
        "tam_estimado": "N/A",
        "trials_recomendados": "N/A",
        "trials_usados": values.get("trials", "N/A"),
    }
    
    version_optimizador = "16"
    
    
    # Generar reporte ASCII (usando los resultados finales)
    reporte_ascii = generar_reporte_ascii(
        values=values,
        config=config,
        best_params=best_params,
        equity_curve=equity_curve_final,
        trades=trades_final,
        version_optimizador=version_optimizador,
        search_space_info=search_space_info,
    )
    
    # ============================================================
    # AGREGAR TABLA DE OVERFITTING SI ESTÁ ACTIVADO
    # ============================================================
    if use_oos:
        tabla_overfitting = generar_tabla_overfitting(
            mejor_corrida=mejor_corrida,
            pf_final=pf_final,
            trades_final=trades_final,
            equity_curve_final=equity_curve_final
        )
        reporte_ascii = reporte_ascii + "\n" + tabla_overfitting
    

    # Ruta del TXT
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
    ruta_txt = os.path.join(output_dir, f"{timestamp} Reporte {safe_symbol}_{timeframe}.txt")
    os.makedirs("reportes", exist_ok=True)
    guardar_reporte_txt(ruta_txt, reporte_ascii)
    
    # Log en consola
    loguear_reporte_en_console(reporte_ascii)
    
    # ============================================================
    # REPORTE DE RENDIMIENTO DETALLADO (usando trades_final)
    # ============================================================
    
    trades = trades_final
    initial_cap = CONSTANTS["initial_capital"]
    final_equity = equity_curve_final.iloc[-1]
    
    # --- PROCESAMIENTO DE ESTADÍSTICAS ---
    longs = [t for t in trades if t['dir'] == "LONG"]
    shorts = [t for t in trades if t['dir'] == "SHORT"]
    
    prof_longs = [t for t in longs if t['net_pnl'] > 0]
    prof_shorts = [t for t in shorts if t['net_pnl'] > 0]
    
    pnl_longs = sum(t['net_pnl'] for t in longs)
    pnl_shorts = sum(t['net_pnl'] for t in shorts)
    
    total_ret_pct = ((final_equity - initial_cap) / initial_cap) * 100
    long_ret_pct = (pnl_longs / initial_cap) * 100
    short_ret_pct = (pnl_shorts / initial_cap) * 100
    win_rate_total = len([t for t in trades if t['net_pnl'] > 0]) / len(trades) * 100 if trades else 0
    max_dd = calcular_drawdown_maximo(list(equity_curve_final))
    
    # Métricas activas en esta corrida
    metricas_activas = []
    if metrics_config.get("use_pf"):       metricas_activas.append(f"PF({metrics_config['peso_pf']:.0f}%)")
    if metrics_config.get("use_winrate"):  metricas_activas.append(f"WinRate({metrics_config['peso_winrate']:.0f}%)")
    if metrics_config.get("use_drawdown"): metricas_activas.append(f"Drawdown({metrics_config['peso_drawdown']:.0f}%)")
    
    print("\n" + "="*50)
    print("         REPORTE DE RENDIMIENTO DETALLADO")
    print("="*50)
    print(f"Símbolo: {symbol} | Timeframe: {timeframe}")
    print(f"Métrica compuesta: {' + '.join(metricas_activas) if metricas_activas else 'PF puro'}")
    print(f"\nRENDIMIENTO TOTAL ESTRATEGIA: {total_ret_pct:+.2f}%")
    print(f"Profit Factor Final:          {pf_final:.2f}")
    print(f"Win Rate Total:               {win_rate_total:.1f}%")
    print(f"Máximo Drawdown:              {max_dd:.2f}%")
    
    print("\n--- DESGLOSE DE OPERACIONES ---")
    print(f"TOTAL TRADES: {len(trades)}")
    print(f"  └─ Total Longs:  {len(longs)}")
    print(f"  └─ Total Shorts: {len(shorts)}")
    
    print(f"\nEFECTIVIDAD (Profitable):")
    win_rate_l = (len(prof_longs) / len(longs) * 100) if longs else 0
    win_rate_s = (len(prof_shorts) / len(shorts) * 100) if shorts else 0
    print(f"  └─ Longs Ganadores:  {len(prof_longs)} ({win_rate_l:.1f}%)")
    print(f"  └─ Shorts Ganadores: {len(prof_shorts)} ({win_rate_s:.1f}%)")
    
    print(f"\nRENDIMIENTO POR LADO:")
    print(f"  └─ Rendimiento% Longs:  {long_ret_pct:.2f}%")
    print(f"  └─ Rendimiento% Shorts: {short_ret_pct:.2f}%")
    
    print("\n--- MEJORES PARÁMETROS ENCONTRADOS ---")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("="*50)
    
    # ============================================================
    # TABLA COMPARATIVA TRAIN vs TEST + FINAL (solo si overfitting activado)
    # ============================================================
    if use_oos:
        print("\n" + "="*65)
        print("  📊 VALIDACIÓN IS/OOS - COMPARATIVA COMPLETA")
        print("="*65)
        print(f"\n{'Métrica':<20} {'TRAIN (70%)':<15} {'TEST (30%)':<15} {'FINAL (100%)':<15}")
        print("-" * 80)
        
        # Profit Factor
        pf_train_val = mejor_corrida.get('pf_train', 0)
        pf_test_val = mejor_corrida.get('pf_test', 0)
        print(f"{'Profit Factor':<20} {pf_train_val:<15.2f} {pf_test_val:<15.2f} {pf_final:<15.2f}")
        
        # Win Rate
        wr_train_val = mejor_corrida.get('winrate_train', 0)
        wr_test_val = mejor_corrida.get('winrate_test', 0)
        winrate_final = len([t for t in trades_final if t['net_pnl'] > 0]) / len(trades_final) * 100 if trades_final else 0
        print(f"{'Win Rate':<20} {wr_train_val:<14.1f}% {wr_test_val:<14.1f}% {winrate_final:<14.1f}%")
        
        # Drawdown
        dd_train_val = mejor_corrida.get('drawdown_train', 0)
        dd_test_val = mejor_corrida.get('drawdown_test', 0)
        drawdown_final = calcular_drawdown_maximo(list(equity_curve_final))
        print(f"{'Drawdown Máx':<20} {dd_train_val:<14.2f}% {dd_test_val:<14.2f}% {drawdown_final:<14.2f}%")
        
        # N° Trades
        trades_train_val = mejor_corrida.get('trades_train', 0)
        trades_test_val = mejor_corrida.get('trades_test', 0)
        print(f"{'N° Trades':<20} {trades_train_val:<15} {trades_test_val:<15} {len(trades_final):<15}")
        
        print("-" * 80)
        
        # Interpretación automática (basada en degradación Train→Test)
        if pf_train_val > 0:
            degradacion_pf = (1 - pf_test_val / pf_train_val) * 100
            if degradacion_pf < 20:
                print("\n✅ ROBUSTO: La degradación del Profit Factor es aceptable (<20%).")
            elif degradacion_pf < 40:
                print("\n⚠️ SOBREAJUSTE MODERADO: La degradación es significativa (20-40%).")
            else:
                print("\n❌ SOBREAJUSTE SEVERO: La estrategia no generaliza (>40% degradación).")
        print("="*65)
    
    # ============================================================
    # 8) GUARDAR TRADES EN CSV (usando trades_final)
    # ============================================================
    if trades_final:
        try:
            df_trades = pd.DataFrame(trades_final)
            
            clean_symbol = symbol.replace("/", "").replace(":", "")
            csv_path = os.path.join(output_dir, f"{timestamp} Trades_csv {clean_symbol}_{timeframe}.csv")
            
            df_trades.to_csv(csv_path, index=False, encoding="utf-8-sig")
            
            print(f"\n[ÉXITO] Archivo de trades generado:")
            print(f"  {csv_path}")
            
        except Exception as e:
            print(f"\n[ERROR] No se pudo guardar el CSV de trades: {e}")
    else:
        print("\n[INFO] No se generaron trades, no se creó el CSV.")
    
    # ============================================================
    # 9) GENERAR SEED JSON
    # ============================================================
    
    metrics = {
        "profit_factor": pf_final,
        "winrate": win_rate_total,
        "drawdown": max_dd,
        "trades": len(trades_final)
    }
    
    output_files = {
        "csv": csv_path if trades_final else None,
        "grafico": None,
        "preview": None,
        "reporte": ruta_txt
    }
    
    guardar_seed(
        values=values,
        config=config,
        best_params=best_params,
        metrics=metrics,
        output_files=output_files,
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp
    )
    
    # ============================================================
    # 10) GRÁFICO INTERACTIVO - Siempre usar datos COMPLETOS
    # ============================================================
    html_path = generar_grafico(df_full, trades_final, best_params, symbol, timeframe)



    
# ============================================================================
# SECCIÓN 18: GENERACIÓN DE GRÁFICOS (PLOTLY)
# ============================================================================


# def generar_grafico movido a charts.py
# def generar_preview_velas movido a charts.py



# ============================================================================
# SECCIÓN 19: ANÁLISIS DE ESPACIO DE BÚSQUEDA
# ============================================================================


def calcular_trials_recomendados(values, config):
    """
    Analiza el espacio de búsqueda activo y sugiere un rango de trials.
    Retorna un string con el análisis completo para mostrar en el output.
    """

    def is_active(key_on, key_auto=None):
        if values.get(key_auto, False):
            return True
        return bool(values.get(key_on, False))

    dimensiones = []

    # --- MAs (siempre activas) ---
    n_tipos_ma = sum(1 for k in ["ma_ema","ma_sma","ma_wma","ma_hma","ma_dema"] if values.get(k))
    rango_ma1 = max(1, config.get("ma1_max", 50) - config.get("ma1_min", 5))
    rango_ma2 = max(1, config.get("ma2_max", 200) - config.get("ma2_min", 20))
    dimensiones.append(("MA1 tipo",    n_tipos_ma,           "categórica"))
    dimensiones.append(("MA1 length",  rango_ma1,            "entera"))
    dimensiones.append(("MA2 tipo",    n_tipos_ma,           "categórica"))
    dimensiones.append(("MA2 length",  rango_ma2,            "entera"))

    # --- RSI ---
    if is_active("use_rsi_long", "auto_rsi_long") or is_active("use_rsi_short", "auto_rsi_short"):
        prefijo = "[AUTO] " if values.get("auto_rsi_long") or values.get("auto_rsi_short") else ""
        rango_rsi_len = max(1, config.get("rsi_length_range", (8, 18))[1] - config.get("rsi_length_range", (8, 18))[0])
        rango_rsi_min = max(1, config.get("rsi_min_range", (55, 65))[1] - config.get("rsi_min_range", (55, 65))[0])
        rango_rsi_max = max(1, config.get("rsi_max_range", (35, 45))[1] - config.get("rsi_max_range", (35, 45))[0])
        dimensiones.append((f"{prefijo}RSI length",  rango_rsi_len, "entera"))
        if is_active("use_rsi_long",  "auto_rsi_long"):
            dimensiones.append((f"{prefijo}RSI min",  rango_rsi_min, "continua"))
        if is_active("use_rsi_short", "auto_rsi_short"):
            dimensiones.append((f"{prefijo}RSI max",  rango_rsi_max, "continua"))
        if values.get("auto_rsi_long") or values.get("auto_rsi_short"):
            dimensiones.append(("[AUTO] RSI activado", 2, "booleana"))

    # --- ADX ---
    if is_active("use_adx_filter", "auto_adx_filter"):
        prefijo = "[AUTO] " if values.get("auto_adx_filter") else ""
        rango_adx_len = max(1, config.get("adx_length_range", (8, 18))[1] - config.get("adx_length_range", (8, 18))[0])
        rango_adx_thr = max(1, config.get("adx_thr_range", (15, 25))[1] - config.get("adx_thr_range", (15, 25))[0])
        dimensiones.append((f"{prefijo}ADX length",    rango_adx_len, "entera"))
        dimensiones.append((f"{prefijo}ADX umbral",    rango_adx_thr, "continua"))
        if values.get("auto_adx_filter"):
            dimensiones.append(("[AUTO] ADX activado", 2, "booleana"))

    # --- High/Low ---
    if is_active("enable_high_condition", "auto_high_condition") or is_active("enable_low_condition", "auto_low_condition"):
        prefijo = "[AUTO] " if values.get("auto_high_condition") or values.get("auto_low_condition") else ""
        rango_lb = max(1, config.get("lookback_range", (2, 10))[1] - config.get("lookback_range", (2, 10))[0])
        dimensiones.append((f"{prefijo}Lookback", rango_lb, "entera"))
        if values.get("auto_high_condition"):
            dimensiones.append(("[AUTO] High activado", 2, "booleana"))
        if values.get("auto_low_condition"):
            dimensiones.append(("[AUTO] Low activado",  2, "booleana"))

    # --- Validation Window ---
    if values.get("use_validation_window"):
        rango_vw = max(1, config.get("valwin_range", (5, 15))[1] - config.get("valwin_range", (5, 15))[0])
        dimensiones.append(("Validation window", rango_vw, "entera"))

    # --- HTF ---
    if is_active("use_htf_filter", "auto_htf_filter"):
        prefijo = "[AUTO] " if values.get("auto_htf_filter") else ""
        rango_htf = max(1, config.get("htf_length_range", (10, 50))[1] - config.get("htf_length_range", (10, 50))[0])
        dimensiones.append((f"{prefijo}HTF length", rango_htf, "entera"))
        if values.get("auto_htf_filter"):
            dimensiones.append(("[AUTO] HTF activado", 2, "booleana"))

    # --- Stop Loss ---
    if is_active("use_stop_loss", "auto_stop_loss"):
        prefijo = "[AUTO] " if values.get("auto_stop_loss") else ""
        rango_sl = max(1, config.get("sl_range", (0.3, 2.0))[1] - config.get("sl_range", (0.3, 2.0))[0])
        dimensiones.append((f"{prefijo}Stop loss %", int(rango_sl * 10), "continua"))
        if values.get("auto_stop_loss"):
            dimensiones.append(("[AUTO] SL activado", 2, "booleana"))

    # --- Break Even ---
    if is_active("activar_stop_be", "auto_stop_be"):
        prefijo = "[AUTO] " if values.get("auto_stop_be") else ""
        rango_be = max(1, config.get("be_range", (1, 10))[1] - config.get("be_range", (1, 10))[0])
        dimensiones.append((f"{prefijo}Velas BE", rango_be, "entera"))
        if values.get("auto_stop_be"):
            dimensiones.append(("[AUTO] BE activado", 2, "booleana"))

    # --- Take Profit ---
    if is_active("use_take_profit_long", "auto_tp_long"):
        prefijo = "[AUTO] " if values.get("auto_tp_long") else ""
        rango_tp = max(1, config.get("tp_long_range", (0.5, 4.0))[1] - config.get("tp_long_range", (0.5, 4.0))[0])
        dimensiones.append((f"{prefijo}TP long %", int(rango_tp * 10), "continua"))
        if values.get("auto_tp_long"):
            dimensiones.append(("[AUTO] TP long activado", 2, "booleana"))

    if is_active("use_take_profit_short", "auto_tp_short"):
        prefijo = "[AUTO] " if values.get("auto_tp_short") else ""
        rango_tp = max(1, config.get("tp_short_range", (0.5, 4.0))[1] - config.get("tp_short_range", (0.5, 4.0))[0])
        dimensiones.append((f"{prefijo}TP short %", int(rango_tp * 10), "continua"))
        if values.get("auto_tp_short"):
            dimensiones.append(("[AUTO] TP short activado", 2, "booleana"))

    # --- Cooldown ---
    if is_active("enable_cooldown", "auto_cooldown"):
        prefijo = "[AUTO] " if values.get("auto_cooldown") else ""
        rango_mls  = max(1, config.get("mls_range",  (1, 3))[1]   - config.get("mls_range",  (1, 3))[0])
        rango_cool = max(1, config.get("cool_range", (10,100))[1]  - config.get("cool_range", (10,100))[0])
        dimensiones.append((f"{prefijo}Max losing streak", rango_mls,  "entera"))
        dimensiones.append((f"{prefijo}Cooldown bars",     rango_cool, "entera"))
        if values.get("auto_cooldown"):
            dimensiones.append(("[AUTO] Cooldown activado", 2, "booleana"))

    # --- Reentry ---
    if is_active("enable_reentry", "auto_reentry"):
        prefijo = "[AUTO] " if values.get("auto_reentry") else ""
        rango_re = max(1, config.get("re_range", (1, 4))[1] - config.get("re_range", (1, 4))[0])
        dimensiones.append((f"{prefijo}Max reentries", rango_re, "entera"))
        if values.get("auto_reentry"):
            dimensiones.append(("[AUTO] Reentry activado", 2, "booleana"))

    if is_active("enable_post_crossover_entry", "auto_post_crossover"):
        prefijo = "[AUTO] " if values.get("auto_post_crossover") else ""
        rango_pr = max(1, config.get("postre_range", (0, 3))[1] - config.get("postre_range", (0, 3))[0])
        dimensiones.append((f"{prefijo}Max post reentries", rango_pr, "entera"))
        if values.get("auto_post_crossover"):
            dimensiones.append(("[AUTO] Post crossover activado", 2, "booleana"))

    # --- Cálculo del espacio y sugerencia ---
    n_dims = len(dimensiones)
    n_autos = sum(1 for nombre, _, _ in dimensiones if nombre.startswith("[AUTO]"))
    import math
    espacio = 1
    for _, rango, _ in dimensiones:
        espacio *= max(2, rango)

    # Lógica de 4 niveles según dimensiones activas
    if n_dims <= 8:
        zona         = "baja"
        t_min, t_max = 600, 1000
        modo_sug     = "Exploración Rápida"
        modo_txt     = "Paralelo"
        multirun     = False
        corridas     = 1
        motivo       = "Espacio chico — una pasada paralela es suficiente para cubrir el espacio."
    elif n_dims <= 12:
        zona         = "media"
        t_min, t_max = 1000, 2000
        modo_sug     = "Exploración Media"
        modo_txt     = "Paralelo"
        multirun     = True
        corridas     = 3
        motivo       = "Espacio moderado — 3 corridas paralelas evalúan impacto de filtros sin convergencia fina."
    elif n_dims <= 18:
        zona         = "media-alta"
        t_min, t_max = 2000, 4000
        modo_sug     = "Converger"
        modo_txt     = "Serie"
        multirun     = True
        corridas     = 3
        motivo       = "Espacio grande — serie profundiza más y 3 corridas verifican estabilidad."
    else:
        zona         = "alta"
        t_min, t_max = 4000, 8000
        modo_sug     = "Validar"
        modo_txt     = "Serie"
        multirun     = True
        corridas     = 5
        motivo       = "Espacio muy grande — serie + 5 corridas para confirmar robustez y evitar falsos óptimos."

    # Si hay features en AUTO, escalar trials hacia el límite superior
    if n_autos > 0:
        t_min = int(t_min * 1.2)
        t_max = int(t_max * 1.3)

    # Redondear a centenas
    t_min = int(round(t_min / 100) * 100)
    t_max = int(round(t_max / 100) * 100)

    multirun_txt = f"ON ({corridas} corridas)" if multirun else "OFF (1 corrida)"

    # --- Armar reporte ---
    lineas = []
    lineas.append("=" * 55)
    lineas.append("  ANÁLISIS DE ESPACIO DE BÚSQUEDA")
    lineas.append("=" * 55)
    lineas.append(f"  Dimensiones activas : {n_dims}  ({n_autos} en modo AUTO)")
    lineas.append(f"  Complejidad         : {zona}")
    lineas.append(f"  Tamaño estimado     : ~{espacio:,.0f} combinaciones")
    lineas.append("")
    lineas.append("  Dimensiones detectadas:")
    for nombre, rango, tipo in dimensiones:
        lineas.append(f"    • {nombre:<32} rango ~{rango:>4}  ({tipo})")
    lineas.append("")
    lineas.append(f"  ► Trials recomendados : {t_min:,} – {t_max:,}")
    if n_autos > 0:
        lineas.append(f"    (escalado +20-30% por {n_autos} feature(s) en AUTO)")
    lineas.append("")
    lineas.append("─" * 55)
    lineas.append("  MODO OPTUNA SUGERIDO")
    lineas.append("─" * 55)
    lineas.append(f"  ► {modo_sug}")
    lineas.append(f"    • Ejecución  : {modo_txt}")
    lineas.append(f"    • Multi-Run  : {multirun_txt}")
    lineas.append(f"    • Motivo     : {motivo}")
    if n_autos > 0:
        lineas.append(f"    • AUTO activo: priorizá el límite superior de trials")
        lineas.append(f"      para que Optuna explore las ramas ON y OFF de cada feature.")
    lineas.append("=" * 55)

    return "\n".join(lineas)



# ============================================================================
# FIN DEL ARCHIVO
# ============================================================================

print("Optimizador cargado correctamente. Ejecutar desde gui.py")