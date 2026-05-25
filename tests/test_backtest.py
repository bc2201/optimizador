"""
Pruebas para el módulo de backtest
"""

import pytest
import pandas as pd
import numpy as np
from backtest import run_backtest, calcular_drawdown_maximo, calcular_score


class TestBacktest:
    """Pruebas para funciones de backtest"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Crea un DataFrame OHLCV de prueba"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        return df
    
    def test_run_backtest_returns_tuple(self, sample_dataframe):
        """Prueba 5: Verificar que run_backtest retorna (profit_factor, equity_curve, trades)"""
        result = run_backtest(
            sample_dataframe,
            ma1_length=10,
            ma2_length=30,
            enable_long_trades=True,
            enable_short_trades=True
        )
        
        assert len(result) == 3  # Tupla de 3 elementos
        assert isinstance(result[0], float)  # profit_factor
        assert isinstance(result[1], pd.Series)  # equity_curve
        assert isinstance(result[2], list)  # trades
    
    def test_drawdown_calculation(self):
        """Prueba 6: Verificar cálculo de drawdown"""
        equity_curve = [1000, 1100, 1050, 1200, 1150, 900, 1100]
        drawdown = calcular_drawdown_maximo(equity_curve)
        
        # El máximo drawdown debe ser desde 1200 hasta 900 = 25%
        assert drawdown == 25.0
    
    def test_score_calculation_with_empty_trades(self):
        """Prueba 7: Score con trades vacíos debe ser 0"""
        metrics_config = {"use_pf": True, "peso_pf": 50}
        score = calcular_score(1.5, [], [], metrics_config)
        assert score == 0.0


    #---------------------------------------------------------------------------------------------------------------

    def test_score_calculation_with_real_trades(self):
        """Prueba 4: calcular_score con trades reales y todas las métricas activas"""
        # Crear trades simulados
        trades = [
            {"net_pnl": 150.0},   # Trade ganador
            {"net_pnl": -50.0},   # Trade perdedor
            {"net_pnl": 200.0},   # Trade ganador
            {"net_pnl": -30.0},   # Trade perdedor
            {"net_pnl": 100.0},   # Trade ganador
        ]
        
        # Curva de equity simulada (capital inicial 1000)
        equity_curve = [1000, 1150, 1100, 1300, 1270, 1370]
        
        # Configuración de métricas con todos los pesos
        metrics_config = {
            "use_pf": True, "peso_pf": 40.0,
            "use_winrate": True, "peso_winrate": 30.0,
            "use_drawdown": True, "peso_drawdown": 20.0,
            "use_n_trades": True, "peso_n_trades": 10.0,
            "min_trades": 3
        }
        
        # Profit Factor = (150+200+100) / (50+30) = 450 / 80 = 5.625
        profit_factor = 5.625
        
        score = calcular_score(profit_factor, trades, equity_curve, metrics_config)
        
        # Verificar que el score está entre 0 y 1 (normalizado)
        assert 0 <= score <= 1
        # Con 3 ganadores de 5, winrate = 60%
        # El score debería ser razonable
        assert score > 0.3
    
    def test_score_calculation_different_weights(self):
        """Prueba 5: Verificar que los pesos afectan el score correctamente"""
        trades = [{"net_pnl": 100} for _ in range(10)]
        equity_curve = [1000 + i*100 for i in range(11)]
        profit_factor = 2.0
        
        # Configuración 1: Solo Profit Factor
        config1 = {"use_pf": True, "peso_pf": 100, "min_trades": 1}
        score1 = calcular_score(profit_factor, trades, equity_curve, config1)
        
        # Configuración 2: Solo Win Rate
        config2 = {"use_winrate": True, "peso_winrate": 100, "min_trades": 1}
        score2 = calcular_score(profit_factor, trades, equity_curve, config2)
        
        # Configuración 3: Solo Profit Factor (con pesos normalizados)
        # El score real depende de la implementación, verificamos consistencia
        print(f"Score1 (solo PF): {score1}")  # Para depuración
        print(f"Score2 (solo WR): {score2}")
        
        # Verificar que ambos scores están en el rango esperado
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        
        # Con winrate=100%, score2 debería ser el máximo (1.0 o cercano)
        # Con profit_factor=2.0, score1 debería ser menor que score2
        assert score2 > score1
    
    def test_score_calculation_min_trades_filter(self):
        """Prueba 6: Verificar que calcular_score NO filtra por min_trades (eso lo hace objective)"""
        trades = [{"net_pnl": 100} for _ in range(5)]
        equity_curve = [1000, 1100, 1200, 1300, 1400, 1500]
        profit_factor = 2.0
        
        config = {
            "use_pf": True, "peso_pf": 50,
            "min_trades": 10  # Este filtro NO aplica aquí
        }
        
        score = calcular_score(profit_factor, trades, equity_curve, config)
        # calcular_score ignora min_trades, siempre calcula score
        assert score > 0  # Debe calcular score aunque haya pocos trades


    #---------------------------------------------------------------------------------------------------------------

    def test_drawdown_multiple_peaks(self):
        """Prueba 7: Drawdown con múltiples picos y valles"""
        # Escenario: sube, baja, sube más, baja más
        equity_curve = [1000, 1200, 1100, 1500, 1300, 1400, 1000]
        
        drawdown = calcular_drawdown_maximo(equity_curve)
        
        # El pico más alto es 1500, el valle más bajo después es 1000
        # Drawdown = (1500 - 1000) / 1500 * 100 = 33.33%
        assert drawdown == 33.33333333333333
    
    def test_drawdown_no_decline(self):
        """Prueba 8: Drawdown cuando nunca baja (equity siempre sube)"""
        equity_curve = [1000, 1100, 1200, 1300, 1400, 1500]
        
        drawdown = calcular_drawdown_maximo(equity_curve)
        
        assert drawdown == 0.0