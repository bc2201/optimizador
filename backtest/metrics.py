"""
Métricas de Rendimiento para Backtest
=====================================
Funciones para calcular:
- Máximo drawdown
- Score compuesto normalizado
"""

import numpy as np





def calcular_drawdown_maximo(equity_curve):
    """
    Calcula el máximo drawdown como porcentaje.
    
    Drawdown = (peak - valor_actual) / peak * 100
    """
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calcular_score(profit_factor, trades, equity_curve, metrics_config):
    if not trades:
        return 0.0

    score = 0.0
    peso_total = 0.0

    if metrics_config.get("use_pf", True):
        peso = metrics_config.get("peso_pf", 50.0)
        pf_norm = min(profit_factor, 10.0) / 10.0
        score += pf_norm * peso
        peso_total += peso

    if metrics_config.get("use_winrate", True):
        peso = metrics_config.get("peso_winrate", 30.0)
        win_rate = len([t for t in trades if t["net_pnl"] > 0]) / len(trades)
        score += win_rate * peso
        peso_total += peso

    if metrics_config.get("use_drawdown", True):
        peso = metrics_config.get("peso_drawdown", 20.0)
        max_dd = calcular_drawdown_maximo(list(equity_curve))
        dd_norm = max(0.0, 1.0 - (max_dd / 100.0))
        score += dd_norm * peso
        peso_total += peso

    if metrics_config.get("use_n_trades", False):
        peso = metrics_config.get("peso_n_trades", 0.0)
        if peso > 0:  # Solo si tiene peso positivo
            n = len(trades)
            min_t = max(1, metrics_config.get("min_trades", 30))
            ref = min_t * 10
            n_norm = min(np.log1p(n / min_t) / np.log1p(ref / min_t), 1.0)
            score += n_norm * peso
            peso_total += peso

    # Si no hay métricas con peso, devolver 0
    if peso_total <= 0:
        return 0.0

    return score / peso_total

