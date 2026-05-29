"""
Reportes ASCII - Generación de reportes en texto plano
=======================================================
Funciones para generar reportes detallados de optimización en formato ASCII.
"""

from datetime import datetime
from backtest import calcular_drawdown_maximo


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


def _bool_mode(values, features, key_flag, key_auto):
    """Devuelve (preset_str, optuna_str) para features booleanos."""
    auto = values.get(key_auto, False)
    base_on = bool(values.get(key_flag, False))

    if auto:
        preset = "AUTO"
        opt_val = features.get(key_flag, None)
        if opt_val is None:
            optuna = "Optuna: N/A"
        else:
            optuna = f"Optuna: {opt_val}"
    else:
        preset = "ON" if base_on else "OFF"
        optuna = "Fijo ON" if base_on else "Fijo OFF"

    return preset, optuna


def _range_str(config_range):
    if not config_range:
        return "min: N/A | max: N/A"
    vmin, vmax = config_range
    return f"min: {vmin} | max: {vmax}"


def generar_reporte_ascii(values, config, best_params, equity_curve, trades, version_optimizador, search_space_info=None):
    """
    Genera reporte con formato unificado para optimización manual.
    """
    now = datetime.now()
    fecha = now.strftime("%d/%m/%Y")
    hora = now.strftime("%H:%M")
    
    # Extraer datos
    symbol = values.get("symbol", "").upper()
    timeframe = values.get("timeframe", "")
    velas = values.get("candles", "")
    
    # ====== MÉTRICAS FINALES ======
    n_trades = len(trades)
    if n_trades == 0:
        profit_factor = 0.0
        total_return = 0.0
        winrate = 0.0
        max_dd = 0.0
        longs_total = shorts_total = longs_win = shorts_win = 0
        longs_ret = shorts_ret = 0.0
    else:
        gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
        gross_loss = -sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0)
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        initial_capital = equity_curve.iloc[0] if len(equity_curve) > 0 else 0.0
        final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0.0
        total_return = ((final_capital / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0

        wins = [t for t in trades if t["net_pnl"] > 0]
        winrate = (len(wins) / n_trades) * 100.0

        max_dd = calcular_drawdown_maximo(list(equity_curve)) if len(equity_curve) > 0 else 0.0

        longs = [t for t in trades if t["dir"] == "LONG"]
        shorts = [t for t in trades if t["dir"] == "SHORT"]
        longs_total = len(longs)
        shorts_total = len(shorts)
        longs_win = len([t for t in longs if t["net_pnl"] > 0])
        shorts_win = len([t for t in shorts if t["net_pnl"] > 0])

        if longs_total > 0:
            cap0 = initial_capital
            cap_long = cap0 + sum(t["net_pnl"] for t in longs)
            longs_ret = ((cap_long / cap0) - 1.0) * 100.0
        else:
            longs_ret = 0.0

        if shorts_total > 0:
            cap0 = initial_capital
            cap_short = cap0 + sum(t["net_pnl"] for t in shorts)
            shorts_ret = ((cap_short / cap0) - 1.0) * 100.0
        else:
            shorts_ret = 0.0

    # ============================================================
    # ENCABEZADO PRINCIPAL
    # ============================================================
    lines = []
    lines.append("=" * 77)
    lines.append(f"🚀 OPTIMIZACIÓN MANUAL\t\t|\tReporte de Optimización")
    lines.append("=" * 77)
    lines.append("")
    lines.append("-" * 77)
    lines.append(f"📅 Fecha: {fecha} {hora}\t-\t🔧 Versión optimizador: {version_optimizador}")
    lines.append("-" * 77)
    lines.append("")
    lines.append(f"📊 Activo: {symbol}\t⏱️ Timeframe: {timeframe}\t📈 Velas: {velas}")
    lines.append("")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Estadísticas del mejor setup encontrado")
    lines.append("-" * 70)
    lines.append(f"Rendimiento total\t= {total_return:+.2f} % \t| Total de operaciones = {n_trades}")
    lines.append("")
    lines.append(f"Profit Factor final \t= {profit_factor:.2f}")
    lines.append(f"Win Rate total\t\t= {winrate:.2f} %")
    lines.append(f"Drawdown\t\t= {max_dd:.2f} %")
    lines.append("")
    lines.append(f"Longs (total)\t= {longs_total}\t| Longs ganadores   =\t{longs_win}\t| Rendimiento longs  = {longs_ret:+.2f} %")
    lines.append(f"Shorts (total)\t= {shorts_total}\t| Shorts ganadores  = \t{shorts_win}\t| Rendimiento shorts = {shorts_ret:+.2f} %")
    lines.append("")
    lines.append("")
    lines.append("📊 Estadísticas de Validación:")
    
    # Score final (si existe en resultado_final, pero en manual no siempre)
    # Buscar score en best_params o calcular algo
    lines.append("")
    lines.append("")
    lines.append("=" * 77)
    lines.append("CONFIGURACIÓN INICIAL")
    lines.append("=" * 77)
    lines.append("")
    
    # ===== MEDIAS MÓVILES =====
    lines.append("| Medias móviles\t |")
    lines.append("+------------------+-----+")
    lines.append(f"| EMA\t\t   | {'ON' if values.get('ma_ema') else 'OFF':<3} |")
    lines.append(f"| SMA\t\t   | {'ON' if values.get('ma_sma') else 'OFF':<3} |")
    lines.append(f"| WMA\t\t   | {'ON' if values.get('ma_wma') else 'OFF':<3} |")
    lines.append(f"| HMA\t\t   | {'ON' if values.get('ma_hma') else 'OFF':<3} |")
    lines.append(f"| DEMA\t\t   | {'ON' if values.get('ma_dema') else 'OFF':<3} |")
    lines.append(f"| MA1 -min\t   | {values.get('ma1_min', ''):<3} |")
    lines.append(f"| MA1 -max\t   | {values.get('ma1_max', ''):<3} |")
    lines.append(f"| MA2 -min\t   | {values.get('ma2_min', ''):<3} |")
    lines.append(f"| MA2 -max\t   | {values.get('ma2_max', ''):<3} |")
    lines.append("+------------------+-----+")
    lines.append("")
    
    # ===== DIRECCIÓN DE TRADES =====
    lines.append("| Dirección de Trades    |")
    lines.append("+------------------+-----+")
    lines.append(f"| Habilitar Longs  | {'ON' if values.get('enable_long_trades') else 'OFF':<3} |")
    lines.append(f"| Habilitar Shorts | {'ON' if values.get('enable_short_trades') else 'OFF':<3} |")
    lines.append("+------------------+-----+")
    lines.append("")
    
    # ===== OPTUNA =====
    modo_opt = values.get("modo_opt", "")
    ejecucion = "Serie" if values.get("modo_serie") else "Paralelo"
    multi_run = "ON" if values.get("multi_run") else "OFF"
    lines.append("| Optuna                                    |")
    lines.append("+----------------------+--------------------+")
    lines.append(f"| Modo                 | {modo_opt:<18} |")
    lines.append(f"| Ejecución            | {ejecucion:<18} |")
    lines.append(f"| Multi-Run            | {multi_run:<18} |")
    lines.append(f"| Cantidad de Corridas | {values.get('multi_runs_count', ''):<18} |")
    lines.append(f"| Cantidad de Trials   | {values.get('trials', ''):<18} |")
    lines.append("+----------------------+--------------------+")
    lines.append("")
    
    # ===== OVERFITTING =====
    use_oos = values.get("use_oos_validation", False)
    lines.append("| Prevención de Overfitting  |")
    lines.append("+-----------------------+----+")
    lines.append(f"| Validación IS/OOS     | {'ON' if use_oos else 'OFF':<3} |")
    lines.append(f"| % Datos Entrenamiento | {values.get('oos_train_pct', '70'):<3} |")
    lines.append("+-----------------------+----+")
    lines.append("")
    
    # ===== MÉTRICAS =====
    lines.append("| Métricas de Optimización (Métrica - On-Off - Peso)|")
    lines.append("+------------------+-----+-----+")
    lines.append(f"| Profit Factor    | {'ON' if values.get('metric_pf') else 'OFF':<3} | {values.get('peso_pf', '50'):>3}% |")
    lines.append(f"| Win Rate         | {'ON' if values.get('metric_winrate') else 'OFF':<3} | {values.get('peso_winrate', '30'):>3}% |")
    lines.append(f"| Max Drawdown     | {'ON' if values.get('metric_drawdown') else 'OFF':<3} | {values.get('peso_drawdown', '20'):>3}% |")
    lines.append(f"| N° de Trades     | {'ON' if values.get('metric_n_trades') else 'OFF':<3} | {values.get('peso_n_trades', '0'):>3}% |")
    lines.append(f"| Mínimo de trades |     | {values.get('min_trades', '30'):>3} |")
    lines.append("+------------------+-----+-----+")
    lines.append("")
    lines.append("=" * 100)
    lines.append("\t\t\tPRESET INICIAL vs FINAL")
    lines.append("=" * 100)
    lines.append("")
    
    # ===== MEDIAS MÓVILES SELECCIONADAS =====
    lines.append("| Medias móviles\t |")
    lines.append("+------------------+-----+")
    lines.append(f"| MA1 -tipo\t   | {best_params.get('ma1_type', 'N/A'):<3} |")
    lines.append(f"| MA1 -longitud\t   | {best_params.get('ma1_length', 'N/A'):<3} |")
    lines.append(f"| MA2 -tipo\t   | {best_params.get('ma2_type', 'N/A'):<3} |")
    lines.append(f"| MA2 -longitud\t   | {best_params.get('ma2_length', 'N/A'):<3} |")
    lines.append("+------------------+-----+")
    lines.append("")
    
    # ===== RSI =====
    preset_rsi_long, opt_rsi_long = _bool_mode(values, best_params, "use_rsi_long", "auto_rsi_long")
    preset_rsi_short, opt_rsi_short = _bool_mode(values, best_params, "use_rsi_short", "auto_rsi_short")
    
    lines.append("| Tendencia (RSI)                                                |")
    lines.append("+-----------------+------------------------------+---------------+")
    lines.append(f"| RSI Long        | {preset_rsi_long:<28} | {opt_rsi_long:<13} |")
    lines.append(f"| RSI Short       | {preset_rsi_short:<28} | {opt_rsi_short:<13} |")
    lines.append(f"| RSI Length      | AUTO ({_range_str(config.get('rsi_length_range')):<20}) | {best_params.get('rsi_length', 'N/A'):<13} |")
    lines.append(f"| RSI min (Long)  | AUTO ({_range_str(config.get('rsi_min_range')):<20}) | {best_params.get('rsi_min', 'N/A'):<13} |")
    lines.append(f"| RSI max (Short) | AUTO ({_range_str(config.get('rsi_max_range')):<20}) | {best_params.get('rsi_max', 'N/A'):<13} |")
    lines.append("+-----------------+------------------------------+---------------+")
    lines.append("")
    
    # ===== ADX =====
    preset_adx, opt_adx = _bool_mode(values, best_params, "use_adx_filter", "auto_adx_filter")
    lines.append("| ADX                                                 |")
    lines.append("+------------+------------------------------+---------+")
    lines.append(f"| ADX        | {preset_adx:<28} | {opt_adx:<7} |")
    lines.append(f"| ADX Length | AUTO ({_range_str(config.get('adx_length_range')):<20}) | {best_params.get('adx_length', 'N/A'):<7} |")
    lines.append(f"| ADX Umbral | AUTO ({_range_str(config.get('adx_thr_range')):<20}) | {best_params.get('adx_threshold', 'N/A'):<7} |")
    lines.append("+------------+------------------------------+---------+")
    lines.append("")
    
    # ===== CONDICIONES DE PRECIO =====
    preset_high, opt_high = _bool_mode(values, best_params, "enable_high_condition", "auto_high_condition")
    preset_low, opt_low = _bool_mode(values, best_params, "enable_low_condition", "auto_low_condition")
    
    lines.append("| Condiciones de Precio                                  |")
    lines.append("+--------------------+-------------------------+---------+")
    lines.append(f"| High Condition     | {preset_high:<23} | {opt_high:<7} |")
    lines.append(f"| Low Condition      | {preset_low:<23} | {opt_low:<7} |")
    lines.append(f"| Lookback           | AUTO ({_range_str(config.get('lookback_range')):<20}) | {best_params.get('lookback', 'N/A'):<7} |")
    val_win_auto = "AUTO" if values.get("use_validation_window") else "OFF"
    val_win_value = best_params.get("validation_window", "N/A") if values.get("use_validation_window") else "Fijo OFF"
    lines.append(f"| Validation Window  | {val_win_auto:<23} | {val_win_value:<7} |")
    lines.append(f"| Range (Val Window) | AUTO ({_range_str(config.get('valwin_range')):<20}) | {best_params.get('validation_window', 'N/A'):<7} |")
    lines.append("+--------------------+-------------------------+---------+")
    lines.append("")
    
    # ===== HTF FILTER =====
    preset_htf, opt_htf = _bool_mode(values, best_params, "use_htf_filter", "auto_htf_filter")
    lines.append("| HTF Filter                                       |")
    lines.append("+------------+--------------------------+----------+")
    lines.append(f"| HTF Filter | {preset_htf:<24} | {opt_htf:<8} |")
    lines.append(f"| Timeframe  | N/A                      | N/A      |")
    lines.append(f"| MA Type    | N/A                      | N/A      |")
    lines.append(f"| HTF Length | AUTO ({_range_str(config.get('htf_length_range')):<20}) | {best_params.get('htf_length', 'N/A'):<8} |")
    lines.append("+------------+--------------------------+----------+")
    lines.append("")
    
    # ===== GESTIÓN DE RIESGO =====
    preset_sl, opt_sl = _bool_mode(values, best_params, "use_stop_loss", "auto_stop_loss")
    preset_be, opt_be = _bool_mode(values, best_params, "activar_stop_be", "auto_stop_be")
    preset_tp_long, opt_tp_long = _bool_mode(values, best_params, "use_take_profit_long", "auto_tp_long")
    preset_tp_short, opt_tp_short = _bool_mode(values, best_params, "use_take_profit_short", "auto_tp_short")
    
    lines.append("| Gestión de Riesgo                                                   |")
    lines.append("+-------------------+----------------------------+--------------------+")
    lines.append(f"| Stop Loss         | {preset_sl:<26} | {opt_sl:<18} |")
    lines.append(f"| SL %              | AUTO ({_range_str(config.get('sl_range')):<20}) | {best_params.get('stop_loss_pct', 'N/A'):<18} |")
    lines.append(f"| Break Even        | {preset_be:<26} | {opt_be:<18} |")
    lines.append(f"| Velas para BE     | AUTO ({_range_str(config.get('be_range')):<20}) | {best_params.get('velas_para_be', 'N/A'):<18} |")
    lines.append(f"| Take Profit Long  | {preset_tp_long:<26} | {opt_tp_long:<18} |")
    lines.append(f"| TP long %         | AUTO ({_range_str(config.get('tp_long_range')):<20}) | {best_params.get('tp_long_pct', 'N/A'):<18} |")
    lines.append(f"| Take Profit Short | {preset_tp_short:<26} | {opt_tp_short:<18} |")
    lines.append(f"| TP short %        | AUTO ({_range_str(config.get('tp_short_range')):<20}) | {best_params.get('tp_short_pct', 'N/A'):<18} |")
    lines.append("+-------------------+----------------------------+--------------------+")
    lines.append("")
    
    # ===== GESTIÓN OPERACIONES =====
    preset_cool, opt_cool = _bool_mode(values, best_params, "enable_cooldown", "auto_cooldown")
    preset_re, opt_re = _bool_mode(values, best_params, "enable_reentry", "auto_reentry")
    preset_post, opt_post = _bool_mode(values, best_params, "enable_post_crossover_entry", "auto_post_crossover")
    
    lines.append("| Gestión Operaciones                                         |")
    lines.append("+----------------------+---------------------------+----------+")
    lines.append(f"| Cooldown             | {preset_cool:<25} | {opt_cool:<8} |")
    lines.append(f"| Max Losing streak    | AUTO ({_range_str(config.get('mls_range')):<20}) | {best_params.get('max_losing_streak', 'N/A'):<8} |")
    lines.append(f"| Cooldown bars        | AUTO ({_range_str(config.get('cool_range')):<20}) | {best_params.get('cooldown_bars', 'N/A'):<8} |")
    lines.append(f"| Reentry              | {preset_re:<25} | {opt_re:<8} |")
    lines.append(f"| Max reentries        | AUTO ({_range_str(config.get('re_range')):<20}) | {best_params.get('max_reentries_allowed', 'N/A'):<8} |")
    lines.append(f"| Post Crossover Entry | {preset_post:<25} | {opt_post:<8} |")
    lines.append(f"| Max post reentries   | AUTO ({_range_str(config.get('postre_range')):<20}) | {best_params.get('max_post_reentries', 'N/A'):<8} |")
    lines.append("+----------------------+---------------------------+----------+")
    lines.append("")
    
    # ===== ESPACIO DE BÚSQUEDA =====
    if search_space_info is None:
        search_space_info = {}
    
    lines.append("=" * 100)
    lines.append("\t\t\tESPACIO DE BÚSQUEDA")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Dimensiones totales \t= {search_space_info.get('dim_totales', 'N/A')}")
    lines.append(f"Complejidad estimada\t= {search_space_info.get('complejidad', 'N/A')}")
    lines.append(f"Tamaño estimado\t\t= {search_space_info.get('tam_estimado', 'N/A')}")
    lines.append(f"Trials recomendados\t= {search_space_info.get('trials_recomendados', 'N/A')}")
    lines.append(f"Trials usados\t\t= {search_space_info.get('trials_usados', values.get('trials', 'N/A'))}")
    lines.append("")
    
    return "\n".join(lines)


def generar_tabla_overfitting(mejor_corrida, pf_final, trades_final, equity_curve_final):
    """Genera la tabla comparativa de overfitting (Train vs Test vs Final)"""
    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("  📊 VALIDACIÓN IS/OOS - COMPARATIVA TRAIN vs TEST vs FINAL")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Métrica':<20} {'TRAIN (70%)':<18} {'TEST (30%)':<18} {'FINAL (100%)':<18}")
    lines.append("-" * 80)
    
    # Profit Factor
    pf_train_val = mejor_corrida.get('pf_train', 0)
    pf_test_val = mejor_corrida.get('pf_test', 0)
    lines.append(f"{'Profit Factor':<20} {pf_train_val:<18.2f} {pf_test_val:<18.2f} {pf_final:<18.2f}")
    
    # Win Rate
    wr_train_val = mejor_corrida.get('winrate_train', 0)
    wr_test_val = mejor_corrida.get('winrate_test', 0)
    winrate_final = len([t for t in trades_final if t['net_pnl'] > 0]) / len(trades_final) * 100 if trades_final else 0
    lines.append(f"{'Win Rate':<20} {wr_train_val:<17.1f}% {wr_test_val:<17.1f}% {winrate_final:<17.1f}%")
    
    # Drawdown
    dd_train_val = mejor_corrida.get('drawdown_train', 0)
    dd_test_val = mejor_corrida.get('drawdown_test', 0)
    drawdown_final = calcular_drawdown_maximo(list(equity_curve_final)) if len(equity_curve_final) > 0 else 0
    lines.append(f"{'Drawdown Máx':<20} {dd_train_val:<17.2f}% {dd_test_val:<17.2f}% {drawdown_final:<17.2f}%")
    
    # N° Trades
    trades_train_val = mejor_corrida.get('trades_train', 0)
    trades_test_val = mejor_corrida.get('trades_test', 0)
    lines.append(f"{'N° Trades':<20} {trades_train_val:<18} {trades_test_val:<18} {len(trades_final):<18}")
    
    lines.append("-" * 80)
    
    # Interpretación automática
    if pf_train_val > 0:
        degradacion_pf = (1 - pf_test_val / pf_train_val) * 100
        if degradacion_pf < 20:
            lines.append("\n✅ ROBUSTO: La degradación del Profit Factor es aceptable (<20%).")
        elif degradacion_pf < 40:
            lines.append("\n⚠️ SOBREAJUSTE MODERADO: La degradación es significativa (20-40%).")
        else:
            lines.append("\n❌ SOBREAJUSTE SEVERO: La estrategia no generaliza (>40% degradación).")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def guardar_reporte_txt(ruta_txt, reporte_ascii):
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(reporte_ascii)


def loguear_reporte_en_console(reporte_ascii, window=None, multiline_key="-OUTPUT-"):
    print(reporte_ascii)
    if window is not None and multiline_key in window.key_dict:
        window[multiline_key].print(reporte_ascii)