"""
Motor de Backtest - Lógica principal de simulación
===================================================
Implementa la estrategia de cruce de medias móviles con múltiples filtros:
- RSI (sobrecompra/sobreventa)
- ADX (fuerza de tendencia)
- High/Lookback (nuevos máximos/mínimos)
- HTF Filter (tendencia de timeframe superior)
- Stop Loss / Take Profit / Break Even
- Cooldown (enfriamiento tras racha de pérdidas)
- Reentradas y post-crossover entries
"""

import numpy as np
import pandas as pd
import threading

# Importaciones desde otros módulos del proyecto
from indicators import ma, rsi_tv, adx_tv
from config import CONSTANTS




def _get_indicator(cache, lock, key, fn):
    """
    Obtiene un indicador del caché de forma thread-safe.
    Útil para evitar recalcular el mismo indicador múltiples veces en paralelo.
    """
    if key in cache:
        return cache[key]
    with lock:
        if key not in cache:
            cache[key] = fn()
        return cache[key]
    


def run_backtest(df,
                 ma1_type="EMA", ma1_length=10,
                 ma2_type="SMA", ma2_length=30,
                 use_rsi_long=True, use_rsi_short=True, rsi_length=14, rsi_min=55.0, rsi_max=45.0,
                 use_adx_filter=True, adx_length=14, adx_threshold=18.0,
                 enable_high_condition=True, enable_low_condition=True, lookback=5,
                 use_validation_window=True, validation_window=20,
                 use_htf_filter=False, htf_tf="1d", htf_type="SMA", htf_length=30,
                 use_stop_loss=True, stop_loss_pct=1.0,
                 activar_stop_be=False, velas_para_be=3,
                 enable_cooldown=True, max_losing_streak=1, cooldown_bars=50,
                 enable_reentry=False, max_reentries_allowed=3,
                 enable_post_crossover_entry=False, max_post_reentries=1,
                 use_take_profit_long=False, tp_long_pct=2.0,
                 use_take_profit_short=True, tp_short_pct=2.0,
                 enable_long_trades=True, enable_short_trades=True,
                 commission_pct=0.075/100.0, initial_capital=2000.0, risk_pct_per_trade=0.30,
                 _cache=None, _lock=None):
    """
    Ejecuta el backtest de la estrategia.
    
    Lógica principal:
        1. Calcula indicadores (MA, RSI, ADX, HTF, High/Low)
        2. Detecta cruces de MA (condición base)
        3. Aplica filtros adicionales (RSI, ADX, etc.)
        4. Gestiona entradas, stops, take profits y reentradas
        5. Registra trades y curva de equity
    
    Returns:
        Tuple (profit_factor, equity_curve, trades_list)
    """
    cache = _cache if _cache is not None else {}
    lock = _lock if _lock is not None else threading.Lock()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- Calcular indicadores (con caché) ---
    ma1_arr = _get_indicator(cache, lock, f"ma1_{ma1_type}_{ma1_length}",
                             lambda: ma(close, ma1_type, ma1_length).to_numpy().reshape(-1))
    ma2_arr = _get_indicator(cache, lock, f"ma2_{ma2_type}_{ma2_length}",
                             lambda: ma(close, ma2_type, ma2_length).to_numpy().reshape(-1))


    if use_rsi_long or use_rsi_short:
        rsi_arr = _get_indicator(cache, lock, f"rsi_{rsi_length}",
                                 lambda: rsi_tv(close, rsi_length).to_numpy().reshape(-1))
    else:
        rsi_arr = None


    if use_adx_filter:
        def _calc_adx():
            adx_s, _, _ = adx_tv(high, low, close, adx_length)
            return adx_s.to_numpy().reshape(-1)
        adx_arr = _get_indicator(cache, lock, f"adx_{adx_length}", _calc_adx)
    else:
        adx_arr = None


    if use_htf_filter:
        def _calc_htf():
            _htf_alias = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                          "1h": "1h", "4h": "4h", "1d": "D", "1w": "W"}
            htf_resampled = close.resample(_htf_alias.get(htf_tf, htf_tf)).last().dropna()
            htf_ma_s = ma(htf_resampled, htf_type, htf_length)
            # Corregido: deprecated method
            return htf_ma_s.reindex(df.index).ffill().to_numpy().reshape(-1)
        htf_arr = _get_indicator(cache, lock, f"htf_{htf_tf}_{htf_type}_{htf_length}", _calc_htf)
    else:
        htf_arr = None


    def _calc_hl_pre():
        high_s = pd.Series(np.asarray(high).ravel(), index=df.index)
        low_s = pd.Series(np.asarray(low).ravel(), index=df.index)
        new_high = high_s.to_numpy() > high_s.shift(1).rolling(lookback).max().to_numpy()
        new_low = low_s.to_numpy() < low_s.shift(1).rolling(lookback).min().to_numpy()
        return (new_high, new_low)

    if enable_high_condition or enable_low_condition:
        hl_pre = _get_indicator(cache, lock, f"hl_pre_{lookback}", _calc_hl_pre)
    else:
        hl_pre = None


    # Arrays base
    close_arr = _get_indicator(cache, lock, "close", lambda: close.to_numpy().reshape(-1))
    high_arr = _get_indicator(cache, lock, "high", lambda: high.to_numpy().reshape(-1))
    low_arr = _get_indicator(cache, lock, "low", lambda: low.to_numpy().reshape(-1))
    index_arr = df.index.to_numpy()


    # --- Limpieza de valores NaN ---
    valid = (~np.isnan(ma1_arr) & ~np.isnan(ma2_arr) & ~np.isnan(close_arr) &
             ~np.isnan(high_arr) & ~np.isnan(low_arr))

    if rsi_arr is not None:
        valid &= ~np.isnan(rsi_arr)
    if adx_arr is not None:
        valid &= ~np.isnan(adx_arr)
    if htf_arr is not None:
        valid &= ~np.isnan(htf_arr)
    if hl_pre is not None:
        valid &= ~np.isnan(hl_pre[0].astype(float))


    # Aplicar máscara
    ma1_arr = ma1_arr[valid]
    ma2_arr = ma2_arr[valid]
    close_arr = close_arr[valid]
    high_arr = high_arr[valid]
    low_arr = low_arr[valid]
    index_arr = index_arr[valid]

    if hl_pre is not None:
        hl_pre_valid = (hl_pre[0][valid], hl_pre[1][valid])
    else:
        hl_pre_valid = None

    if rsi_arr is not None:
        rsi_arr = rsi_arr[valid]
    if adx_arr is not None:
        adx_arr = adx_arr[valid]
    if htf_arr is not None:
        htf_arr = htf_arr[valid]

    n = len(close_arr)


    # --- Condiciones vectorizadas ---
    ma_cross_long = np.zeros(n, dtype=bool)
    ma_cross_short = np.zeros(n, dtype=bool)
    ma_cross_long[1:] = (ma1_arr[1:] > ma2_arr[1:]) & (ma1_arr[:-1] <= ma2_arr[:-1])
    ma_cross_short[1:] = (ma1_arr[1:] < ma2_arr[1:]) & (ma1_arr[:-1] >= ma2_arr[:-1])

    price_above = (close_arr > ma1_arr) & (close_arr > ma2_arr)
    price_below = (close_arr < ma1_arr) & (close_arr < ma2_arr)

    rsi_above = (rsi_arr > rsi_min) if use_rsi_long else np.ones(n, dtype=bool)
    rsi_below = (rsi_arr < rsi_max) if use_rsi_short else np.ones(n, dtype=bool)

    if enable_high_condition or enable_low_condition:
        new_high, new_low = hl_pre_valid
        high_cond = new_high if enable_high_condition else np.ones(n, dtype=bool)
        low_cond = new_low if enable_low_condition else np.ones(n, dtype=bool)
    else:
        high_cond = np.ones(n, dtype=bool)
        low_cond = np.ones(n, dtype=bool)

    adx_filt = (adx_arr > adx_threshold) if use_adx_filter else np.ones(n, dtype=bool)
    htf_long = (close_arr > htf_arr) if use_htf_filter else np.ones(n, dtype=bool)
    htf_short = (close_arr < htf_arr) if use_htf_filter else np.ones(n, dtype=bool)

    cond_long_arr = price_above & rsi_above & high_cond & htf_long & adx_filt
    cond_short_arr = price_below & rsi_below & low_cond & htf_short & adx_filt

    # --- Variables de estado del backtest ---
    long_cross_bar = np.nan
    short_cross_bar = np.nan
    position = 0
    entry_price = np.nan
    entry_equity = initial_capital
    entry_time = None
    stop_price = np.nan
    tp_price = np.nan
    velas_desde_entrada = 0
    precio_stop_be = np.nan
    tipo_operacion = ""
    ultima_operacion_cerrada = ""
    reentry_bar_index = np.nan
    reentries_count = 0
    ultima_direccion_ma = ""
    post_reentries_count = 0
    losing_streak = 0
    cooldown_start_bar = np.nan
    in_cooldown = False
    equity = initial_capital
    equity_curve = []
    trades = []
    ultima_entrada_long_bar = np.nan
    ultima_entrada_short_bar = np.nan
    bar_index = 0

    # --- Bucle principal (simulación barra por barra) ---
    for i in range(n):
        bar_index += 1
        idx = index_arr[i]

        close_i = close_arr[i]
        high_i = high_arr[i]
        low_i = low_arr[i]
        ma2_i = ma2_arr[i]
        crossover_i = ma_cross_long[i]
        crossunder_i = ma_cross_short[i]
        cond_long_i = cond_long_arr[i]
        cond_short_i = cond_short_arr[i]
        ma1_i = ma1_arr[i]

        # Actualizar barras de cruce
        if crossover_i:
            long_cross_bar = bar_index
            reentries_count = 0
            ultima_operacion_cerrada = ""
            ultima_direccion_ma = "long"
            post_reentries_count = 0

        if crossunder_i:
            short_cross_bar = bar_index
            reentries_count = 0
            ultima_operacion_cerrada = ""
            ultima_direccion_ma = "short"
            post_reentries_count = 0

        # Ventanas de validación
        long_window_active = (use_validation_window and not np.isnan(long_cross_bar) and
                              (bar_index - long_cross_bar <= validation_window))
        short_window_active = (use_validation_window and not np.isnan(short_cross_bar) and
                               (bar_index - short_cross_bar <= validation_window))

        # Cooldown (enfriamiento tras racha de pérdidas)
        if enable_cooldown and not np.isnan(cooldown_start_bar):
            in_cooldown = (bar_index - cooldown_start_bar) < cooldown_bars
        else:
            in_cooldown = False
        allow_trade = not in_cooldown

        enter_long_base = long_window_active if use_validation_window else crossover_i
        enter_short_base = short_window_active if use_validation_window else crossunder_i

        # Evitar reentradas en la misma ventana si reentry está desactivado
        already_entered_in_window_long = ((not enable_reentry) and (not enable_post_crossover_entry) and
                                          use_validation_window and (not np.isnan(ultima_entrada_long_bar)) and
                                          (bar_index - ultima_entrada_long_bar <= validation_window))
        already_entered_in_window_short = ((not enable_reentry) and (not enable_post_crossover_entry) and
                                           use_validation_window and (not np.isnan(ultima_entrada_short_bar)) and
                                           (bar_index - ultima_entrada_short_bar <= validation_window))

        # Condiciones de reentrada
        reentry_long_active = (enable_reentry and ultima_operacion_cerrada == "long" and cond_long_i and
                               allow_trade and (reentries_count < max_reentries_allowed) and
                               (not np.isnan(reentry_bar_index)) and (bar_index > reentry_bar_index) and
                               enable_long_trades and tipo_operacion == "")

        reentry_short_active = (enable_reentry and ultima_operacion_cerrada == "short" and cond_short_i and
                                allow_trade and (reentries_count < max_reentries_allowed) and
                                (not np.isnan(reentry_bar_index)) and (bar_index > reentry_bar_index) and
                                enable_short_trades and tipo_operacion == "")

        # Entradas post-crossover
        post_crossover_long = (enable_post_crossover_entry and ultima_operacion_cerrada == "" and
                               ultima_direccion_ma == "long" and (ma1_i > ma2_i) and cond_long_i and
                               allow_trade and (post_reentries_count < max_post_reentries) and
                               enable_long_trades and tipo_operacion == "")

        post_crossover_short = (enable_post_crossover_entry and ultima_operacion_cerrada == "" and
                                ultima_direccion_ma == "short" and (ma1_i < ma2_i) and cond_short_i and
                                allow_trade and (post_reentries_count < max_post_reentries) and
                                enable_short_trades and tipo_operacion == "")

        # Determinar entrada
        enter_long = False
        enter_short = False

        if (allow_trade and enter_long_base and cond_long_i and enable_long_trades and
            tipo_operacion == "" and not already_entered_in_window_long):
            enter_long = True
        elif reentry_long_active:
            enter_long = True
        elif post_crossover_long:
            enter_long = True

        if (allow_trade and enter_short_base and cond_short_i and enable_short_trades and
            tipo_operacion == "" and not already_entered_in_window_short):
            enter_short = True
        elif reentry_short_active:
            enter_short = True
        elif post_crossover_short:
            enter_short = True

        # Contar reentradas
        if reentry_long_active or reentry_short_active:
            reentries_count += 1
        if post_crossover_long or post_crossover_short:
            post_reentries_count += 1

        # Gestión de posición abierta (Break Even)
        if position != 0:
            velas_desde_entrada += 1
            if activar_stop_be and velas_desde_entrada >= velas_para_be:
                precio_stop_be = entry_price
            else:
                precio_stop_be = np.nan
        else:
            velas_desde_entrada = 0
            precio_stop_be = np.nan
            tp_price = np.nan
            stop_price = np.nan
            tipo_operacion = ""

        # --- Cierres de posición ---
        if position != 0:
            if position > 0:  # LONG
                # Calcular SL y TP
                if use_stop_loss:
                    if not np.isnan(precio_stop_be):
                        stop_price = precio_stop_be
                    else:
                        stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                if use_take_profit_long and np.isnan(tp_price):
                    tp_price = entry_price * (1 + tp_long_pct / 100.0)

                hit_stop = use_stop_loss and (low_i <= stop_price)
                hit_tp = use_take_profit_long and (high_i >= tp_price)
                exit_long_manual = (close_i < ma2_i)

                if hit_stop or hit_tp or exit_long_manual:
                    # Determinar precio y razón de salida
                    if hit_stop and hit_tp:
                        exit_reason = "stop"
                        exit_price = stop_price
                    elif hit_stop:
                        exit_reason = "stop"
                        exit_price = stop_price
                    elif hit_tp:
                        exit_reason = "tp"
                        exit_price = tp_price
                    else:
                        exit_reason = "manual"
                        exit_price = close_i

                    # Calcular PnL
                    size = (entry_equity * risk_pct_per_trade) / entry_price
                    gross_pnl = (exit_price - entry_price) * size
                    commission = (entry_price + exit_price) * size * commission_pct
                    net_pnl = gross_pnl - commission
                    equity += net_pnl

                    trades.append({
                        "dir": "LONG",
                        "entry_time": entry_time,
                        "exit_time": idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": size,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "reason": exit_reason
                    })

                    # Actualizar racha de pérdidas
                    if net_pnl < 0:
                        losing_streak += 1
                    else:
                        losing_streak = 0

                    # Activar cooldown si corresponde
                    if enable_cooldown and losing_streak >= max_losing_streak:
                        cooldown_start_bar = bar_index
                        losing_streak = 0

                    # Resetear estado
                    ultima_operacion_cerrada = "long"
                    reentry_bar_index = bar_index
                    position = 0
                    tipo_operacion = ""
                    velas_desde_entrada = 0
                    precio_stop_be = np.nan
                    tp_price = np.nan
                    stop_price = np.nan

            elif position < 0:  # SHORT (lógica simétrica)
                if use_stop_loss:
                    if not np.isnan(precio_stop_be):
                        stop_price = precio_stop_be
                    else:
                        stop_price = entry_price * (1 + stop_loss_pct / 100.0)
                if use_take_profit_short and np.isnan(tp_price):
                    tp_price = entry_price * (1 - tp_short_pct / 100.0)

                hit_stop = use_stop_loss and (high_i >= stop_price)
                hit_tp = use_take_profit_short and (low_i <= tp_price)
                exit_short_manual = (close_i > ma2_i)

                if hit_stop or hit_tp or exit_short_manual:
                    if hit_stop and hit_tp:
                        exit_reason = "stop"
                        exit_price = stop_price
                    elif hit_stop:
                        exit_reason = "stop"
                        exit_price = stop_price
                    elif hit_tp:
                        exit_reason = "tp"
                        exit_price = tp_price
                    else:
                        exit_reason = "manual"
                        exit_price = close_i

                    size = (entry_equity * risk_pct_per_trade) / entry_price
                    gross_pnl = (entry_price - exit_price) * size
                    commission = (entry_price + exit_price) * size * commission_pct
                    net_pnl = gross_pnl - commission
                    equity += net_pnl

                    trades.append({
                        "dir": "SHORT",
                        "entry_time": entry_time,
                        "exit_time": idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": size,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "reason": exit_reason
                    })

                    if net_pnl < 0:
                        losing_streak += 1
                    else:
                        losing_streak = 0

                    if enable_cooldown and losing_streak >= max_losing_streak:
                        cooldown_start_bar = bar_index
                        losing_streak = 0

                    ultima_operacion_cerrada = "short"
                    reentry_bar_index = bar_index
                    position = 0
                    tipo_operacion = ""
                    velas_desde_entrada = 0
                    precio_stop_be = np.nan
                    tp_price = np.nan
                    stop_price = np.nan

        # --- Entradas nuevas ---
        if position == 0:
            if enter_long:
                position = 1
                entry_price = close_i
                entry_equity = equity
                entry_time = idx
                tipo_operacion = "long"
                velas_desde_entrada = 0
                reentry_bar_index = np.nan
                ultima_entrada_long_bar = bar_index

                if use_take_profit_long:
                    tp_price = entry_price * (1 + tp_long_pct / 100.0)
                if use_stop_loss:
                    stop_price = entry_price * (1 - stop_loss_pct / 100.0)

            elif enter_short:
                position = -1
                entry_price = close_i
                entry_equity = equity
                entry_time = idx
                tipo_operacion = "short"
                velas_desde_entrada = 0
                reentry_bar_index = np.nan
                ultima_entrada_short_bar = bar_index

                if use_take_profit_short:
                    tp_price = entry_price * (1 - tp_short_pct / 100.0)
                if use_stop_loss:
                    stop_price = entry_price * (1 + stop_loss_pct / 100.0)

        equity_curve.append(equity)

    # --- Calcular métricas finales ---
    if len(trades) == 0:
        return 0.0, pd.Series(equity_curve, index=index_arr[:len(equity_curve)]), trades

    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = -sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0)

    if gross_loss == 0:
        profit_factor = np.inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    return profit_factor, pd.Series(equity_curve, index=index_arr[:len(equity_curve)]), trades