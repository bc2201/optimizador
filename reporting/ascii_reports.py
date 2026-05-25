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
    """
    Devuelve (preset_str, optuna_str) para features booleanos tipo:
    - use_rsi_long, auto_rsi_long
    - use_adx_filter, auto_adx_filter
    """
    auto = values.get(key_auto, False)
    base_on = bool(values.get(key_flag, False))

    if auto:
        preset = "AUTO"
        # si estuvo en AUTO, Optuna devuelve un bool en cleaned_best_params
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
    best_params: cleaned_best_params de tu run_optuna_with_gui (params finales usados en backtest).
    equity_curve: Serie de equity final (df_test).
    trades: lista de trades final (df_test).
    search_space_info: dict opcional con info del espacio de búsqueda.
    """

    now = datetime.now()
    fecha = now.strftime("%d/%m/%Y")
    hora = now.strftime("%H:%M")

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

    header = []
    header.append("="*100)
    header.append("\t\t\tREPORTE DE RENDIMIENTO DETALLADO")
    header.append("="*100)
    header.append("")
    header.append(f"Fecha de corrida = {fecha}\t\t| Hora= {hora}\t\t| Versión del optimizador = {version_optimizador}")
    header.append("")
    header.append(f"Rendimiento total\t= {total_return:.2f} % \t| Total de operaciones = {n_trades}")
    header.append("")
    header.append(f"Profit Factor final \t= {profit_factor:.2f}")
    header.append(f"Win Rate total\t\t= {winrate:.2f} %")
    header.append(f"Drawdown\t\t= {max_dd:.2f} %")
    header.append("")
    header.append(f"Longs (total)\t= {longs_total}\t| Longs ganadores   =\t{longs_win}\t| Rendimiento longs  = {longs_ret:.2f} %")
    header.append(f"Shorts (total)\t= {shorts_total}\t| Shorts ganadores  = \t{shorts_win}\t| Rendimiento shorts = {shorts_ret:.2f} %")
    header.append("")
    header.append("")

    bloques = []
    bloques.append("="*100)
    bloques.append("\t\t\tCONFIGURACIÓN INICIAL")
    bloques.append("="*100)
    bloques.append("")

    # ===== CONFIGURACIÓN DEL ACTIVO =====
    rows_activo = [
        ("Símbolo",   values.get("symbol", "").upper()),
        ("Timeframe", values.get("timeframe", "")),
        ("Velas",     values.get("candles", "")),
    ]
    bloques.append(build_ascii_table(rows_activo, "Configuración del Activo"))
    bloques.append("")

    # ===== DIRECCIÓN DE TRADES =====
    rows_dir = [
        ("Habilitar Longs",  "ON" if values.get("enable_long_trades") else "OFF"),
        ("Habilitar Shorts", "ON" if values.get("enable_short_trades") else "OFF"),
    ]
    bloques.append(build_ascii_table(rows_dir, "Dirección de Trades"))
    bloques.append("")

    # ===== OPTUNA =====
    modo_opt = values.get("modo_opt", "")
    ejecucion = "Serie" if values.get("modo_serie") else "Paralelo"
    multi_run = "ON" if values.get("multi_run") else "OFF"
    rows_optuna = [
        ("Modo",                 modo_opt),
        ("Ejecución",            ejecucion),
        ("Multi-Run",            multi_run),
        ("Cantidad de Corridas", values.get("multi_runs_count", "")),
        ("Cantidad de Trials",   values.get("trials", "")),
    ]
    bloques.append(build_ascii_table(rows_optuna, "Optuna"))
    bloques.append("")

    # ===== OVERFITTING =====
    rows_oof = [
        ("Validación IS/OOS",     "ON" if values.get("use_oos_validation") else "OFF"),
        ("% Datos Entrenamiento", values.get("oos_train_pct", "")),
    ]
    bloques.append(build_ascii_table(rows_oof, "Prevención de Overfitting"))
    bloques.append("")

    # ===== MÉTRICAS DE OPTIMIZACIÓN (usás config fijo en GUI) =====
    rows_metricas = [
        ("Profit Factor",   "ON",  f"{values.get('peso_pf', '50')}%"),
        ("Win Rate",        "ON",  f"{values.get('peso_winrate', '30')}%"),
        ("Max Drawdown",    "ON",  f"{values.get('peso_drawdown', '20')}%"),
        ("N° de Trades",    "ON" if values.get("metric_n_trades", False) else "OFF", f"{values.get('peso_n_trades', '0')}%"),
        ("Mínimo de trades","",    f"{values.get('min_trades', '30')}"),
    ]
    bloques.append(build_ascii_table(rows_metricas, "Métricas de Optimización (Métrica - On-Off - Peso)"))
    bloques.append("")

    # ============================================================
    # PRESET INICIAL vs FINAL
    # ============================================================
    bloques.append("="*100)
    bloques.append("\t\t\tPRESET INICIAL vs FINAL")
    bloques.append("="*100)
    bloques.append("")

    # ===== TENDENCIA (RSI) =====
    # modos booleanos (AUTO con rangos)
    preset_rsi_long, opt_rsi_long = _bool_mode(values, best_params, "use_rsi_long", "auto_rsi_long")
    preset_rsi_short, opt_rsi_short = _bool_mode(values, best_params, "use_rsi_short", "auto_rsi_short")

    rows_rsi = [
        ("RSI Long",  preset_rsi_long,  opt_rsi_long),
        ("RSI Short", preset_rsi_short, opt_rsi_short),
        (
            "RSI Length",
            f"AUTO ({_range_str(config.get('rsi_length_range'))})",
            best_params.get("rsi_length", "N/A")
        ),
        (
            "RSI min (Long)",
            f"AUTO ({_range_str(config.get('rsi_min_range'))})",
            best_params.get("rsi_min", "N/A")
        ),
        (
            "RSI max (Short)",
            f"AUTO ({_range_str(config.get('rsi_max_range'))})",
            best_params.get("rsi_max", "N/A")
        ),
    ]
    bloques.append(build_ascii_table(rows_rsi, "Tendencia (RSI)"))
    bloques.append("")

    # ===== ADX =====
    preset_adx, opt_adx = _bool_mode(values, best_params, "use_adx_filter", "auto_adx_filter")
    rows_adx = [
        ("ADX",          preset_adx, opt_adx),
        (
            "ADX Length",
            f"AUTO ({_range_str(config.get('adx_length_range'))})",
            best_params.get("adx_length", "N/A")
        ),
        (
            "ADX Umbral",
            f"AUTO ({_range_str(config.get('adx_thr_range'))})",
            best_params.get("adx_threshold", "N/A")
        ),
    ]
    bloques.append(build_ascii_table(rows_adx, "ADX"))
    bloques.append("")

    # ===== CONDICIONES DE PRECIO =====
    preset_high, opt_high = _bool_mode(values, best_params, "enable_high_condition", "auto_high_condition")
    preset_low, opt_low   = _bool_mode(values, best_params, "enable_low_condition", "auto_low_condition")

    rows_price = [
        ("High Condition", preset_high, opt_high),
        ("Low Condition",  preset_low,  opt_low),
        (
            "Lookback",
            f"AUTO ({_range_str(config.get('lookback_range'))})",
            best_params.get("lookback", "N/A")
        ),
        (
            "Validation Window",
            "AUTO" if values.get("use_validation_window") else "OFF",
            best_params.get("validation_window", "N/A") if values.get("use_validation_window") else "Fijo OFF"
        ),
        (
            "Range (Val Window)",
            f"AUTO ({_range_str(config.get('valwin_range'))})",
            best_params.get("validation_window", "N/A") if values.get("use_validation_window") else "N/A"
        ),
    ]
    bloques.append(build_ascii_table(rows_price, "Condiciones de Precio"))
    bloques.append("")

    # ===== HTF FILTER =====
    preset_htf, opt_htf = _bool_mode(values, best_params, "use_htf_filter", "auto_htf_filter")
    rows_htf = [
        ("HTF Filter", preset_htf, opt_htf),
        ("Timeframe",  values.get("htf_timeframe", "N/A"), best_params.get("htf_tf", "N/A")),
        ("MA Type",    values.get("htf_ma_type", "N/A"),   best_params.get("htf_type", "N/A")),
        (
            "HTF Length",
            f"AUTO ({_range_str(config.get('htf_length_range'))})",
            best_params.get("htf_length", "N/A")
        ),
    ]
    bloques.append(build_ascii_table(rows_htf, "HTF Filter"))
    bloques.append("")

    # ===== GESTIÓN DE RIESGO =====
    preset_sl, opt_sl = _bool_mode(values, best_params, "use_stop_loss", "auto_stop_loss")
    preset_be, opt_be = _bool_mode(values, best_params, "activar_stop_be", "auto_stop_be")
    preset_tp_long, opt_tp_long = _bool_mode(values, best_params, "use_take_profit_long", "auto_tp_long")
    preset_tp_short, opt_tp_short = _bool_mode(values, best_params, "use_take_profit_short", "auto_tp_short")

    rows_risk = [
        ("Stop Loss", preset_sl, opt_sl),
        (
            "SL %",
            f"AUTO ({_range_str(config.get('sl_range'))})",
            best_params.get("stop_loss_pct", "N/A")
        ),
        ("Break Even", preset_be, opt_be),
        (
            "Velas para BE",
            f"AUTO ({_range_str(config.get('be_range'))})",
            best_params.get("velas_para_be", "N/A")
        ),
        ("Take Profit Long", preset_tp_long, opt_tp_long),
        (
            "TP long %",
            f"AUTO ({_range_str(config.get('tp_long_range'))})",
            best_params.get("tp_long_pct", "N/A")
        ),
        ("Take Profit Short", preset_tp_short, opt_tp_short),
        (
            "TP short %",
            f"AUTO ({_range_str(config.get('tp_short_range'))})",
            best_params.get("tp_short_pct", "N/A")
        ),
    ]
    bloques.append(build_ascii_table(rows_risk, "Gestión de Riesgo"))
    bloques.append("")

    # ===== GESTIÓN OPERACIONES =====
    preset_cool, opt_cool = _bool_mode(values, best_params, "enable_cooldown", "auto_cooldown")
    preset_re, opt_re = _bool_mode(values, best_params, "enable_reentry", "auto_reentry")
    preset_post, opt_post = _bool_mode(values, best_params, "enable_post_crossover_entry", "auto_post_crossover")

    rows_ops = [
        ("Cooldown", preset_cool, opt_cool),
        (
            "Max Losing streak",
            f"AUTO ({_range_str(config.get('mls_range'))})",
            best_params.get("max_losing_streak", "N/A")
        ),
        (
            "Cooldown bars",
            f"AUTO ({_range_str(config.get('cool_range'))})",
            best_params.get("cooldown_bars", "N/A")
        ),
        ("Reentry", preset_re, opt_re),
        (
            "Max reentries",
            f"AUTO ({_range_str(config.get('re_range'))})",
            best_params.get("max_reentries_allowed", "N/A")
        ),
        ("Post Crossover Entry", preset_post, opt_post),
        (
            "Max post reentries",
            f"AUTO ({_range_str(config.get('postre_range'))})",
            best_params.get("max_post_reentries", "N/A")
        ),
    ]
    bloques.append(build_ascii_table(rows_ops, "Gestión Operaciones"))
    bloques.append("")

    # ===== ESPACIO DE BÚSQUEDA =====
    bloques.append("="*100)
    bloques.append("\t\t\tESPACIO DE BÚSQUEDA")
    bloques.append("="*100)
    bloques.append("")

    if search_space_info is None:
        search_space_info = {}

    bloques.append(f"Dimensiones totales \t= {search_space_info.get('dim_totales', 'N/A')}")
    bloques.append(f"Complejidad estimada\t= {search_space_info.get('complejidad', 'N/A')}")
    bloques.append(f"Tamaño estimado\t\t= {search_space_info.get('tam_estimado', 'N/A')}")
    bloques.append(f"Trials recomendados\t= {search_space_info.get('trials_recomendados', 'N/A')}")
    bloques.append(f"Trials usados\t\t= {search_space_info.get('trials_usados', values.get('trials', 'N/A'))}")
    bloques.append("")

    reporte = "\n".join(header + bloques)
    return reporte


def guardar_reporte_txt(ruta_txt, reporte_ascii):
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(reporte_ascii)


def loguear_reporte_en_console(reporte_ascii, window=None, multiline_key="-OUTPUT-"):
    print(reporte_ascii)
    if window is not None and multiline_key in window.key_dict:  # <--- CORRECCIÓN: 'key_dict'
        window[multiline_key].print(reporte_ascii)
