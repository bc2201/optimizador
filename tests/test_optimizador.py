"""
Pruebas para optimizador_main.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock


class TestOptimizador:
    """Pruebas para funciones de optimizador_main"""
    
    @pytest.fixture
    def sample_values(self):
        """Crea valores simulados del GUI"""
        return {
            "ma_ema": True,
            "ma_sma": True,
            "ma1_min": "5",
            "ma1_max": "20",
            "ma2_min": "20",
            "ma2_max": "100",
            "rsi_length_min": "8",
            "rsi_length_max": "18",
            "rsi_min_min": "55",
            "rsi_min_max": "65",
            "rsi_max_min": "35",
            "rsi_max_max": "45",
            "adx_length_min": "8",
            "adx_length_max": "18",
            "adx_threshold_min": "15",
            "adx_threshold_max": "25",
            "lookback_min": "2",
            "lookback_max": "10",
            "validation_window_min": "5",
            "validation_window_max": "15",
            "htf_length_min": "10",
            "htf_length_max": "50",
            "stop_loss_min": "0.3",
            "stop_loss_max": "2.0",
            "velas_para_be_min": "1",
            "velas_para_be_max": "10",
            "tp_long_min": "0.5",
            "tp_long_max": "4.0",
            "tp_short_min": "0.5",
            "tp_short_max": "4.0",
            "max_losing_streak_min": "1",
            "max_losing_streak_max": "3",
            "cooldown_bars_min": "10",
            "cooldown_bars_max": "100",
            "max_reentries_min": "1",
            "max_reentries_max": "4",
            "max_post_reentries_min": "0",
            "max_post_reentries_max": "3",
        }
    
    def test_get_range_valid_values(self):
        """Prueba 1: get_range con valores válidos"""
        from optimizador_main import get_range
        
        values = {"min": "10", "max": "20"}
        result_min, result_max = get_range(values, "min", "max", 0, 100, int)
        
        assert result_min == 10
        assert result_max == 20
    
    def test_get_range_inverted_values(self):
        """Prueba 2: get_range con min > max (debe devolver defaults)"""
        from optimizador_main import get_range
        
        values = {"min": "20", "max": "10"}
        result_min, result_max = get_range(values, "min", "max", 0, 100, int)
        
        # min > max, debe retornar defaults
        assert result_min == 0
        assert result_max == 100
    
    def test_get_range_empty_values(self):
        """Prueba 3: get_range con valores vacíos (debe devolver defaults)"""
        from optimizador_main import get_range
        
        values = {"min": "", "max": ""}
        result_min, result_max = get_range(values, "min", "max", 5, 50, int)
        
        assert result_min == 5
        assert result_max == 50
    
    def test_build_config_valid(self, sample_values):
        """Prueba 4: build_config con valores válidos"""
        from optimizador_main import build_config
        
        config = build_config(sample_values)
        
        assert config is not None
        assert "tipos_ma" in config
        assert len(config["tipos_ma"]) >= 1
        assert "ma1_min" in config
        assert "ma2_min" in config
        assert "rsi_length_range" in config
    
    def test_build_config_no_ma_selected(self):
        """Prueba 5: build_config sin ningún tipo de MA seleccionado"""
        from optimizador_main import build_config
        
        values = {
            "ma_ema": False,
            "ma_sma": False,
            "ma_wma": False,
            "ma_hma": False,
            "ma_dema": False,
        }
        
        config = build_config(values)
        assert config is None  # Debe retornar None
    
    def test_calcular_trials_recomendados_basic(self):
        """Prueba 6: calcular_trials_recomendados con configuración básica"""
        from optimizador_main import calcular_trials_recomendados
        
        # Valores mínimos para simular configuración
        values = {
            "ma_ema": True, "ma_sma": True,
            "ma1_min": "5", "ma1_max": "20",
            "ma2_min": "20", "ma2_max": "100",
            "use_rsi_long": False, "use_rsi_short": False,
            "use_adx_filter": False,
            "enable_high_condition": True, "enable_low_condition": True,
            "use_validation_window": True,
            "use_htf_filter": False, "use_stop_loss": False,
            "activar_stop_be": False, "enable_cooldown": False,
            "enable_reentry": False, "enable_post_crossover_entry": False,
            "use_take_profit_long": False, "use_take_profit_short": False,
            "auto_rsi_long": False, "auto_rsi_short": False,
            "auto_adx_filter": False, "auto_htf_filter": False,
            "auto_stop_loss": False, "auto_stop_be": False,
            "auto_tp_long": False, "auto_tp_short": False,
            "auto_cooldown": False, "auto_reentry": False,
            "auto_post_crossover": False
        }
        
        # Config mock
        config = {
            "ma1_min": 5, "ma1_max": 20,
            "ma2_min": 20, "ma2_max": 100,
            "rsi_length_range": (8, 18),
            "rsi_min_range": (55, 65),
            "rsi_max_range": (35, 45),
            "adx_length_range": (8, 18),
            "adx_thr_range": (15, 25),
            "lookback_range": (2, 10),
            "valwin_range": (5, 15),
            "htf_length_range": (10, 50),
            "sl_range": (0.3, 2.0),
            "be_range": (1, 10),
            "tp_long_range": (0.5, 4.0),
            "tp_short_range": (0.5, 4.0),
            "mls_range": (1, 3),
            "cool_range": (10, 100),
            "re_range": (1, 4),
            "postre_range": (0, 3),
        }
        
        reporte = calcular_trials_recomendados(values, config)
        
        # Debe retornar un string no vacío
        assert isinstance(reporte, str)
        assert len(reporte) > 0
        assert "ANÁLISIS DE ESPACIO DE BÚSQUEDA" in reporte
    
    @patch('optimizador_main.fetch_binance_ohlcv')
    def test_get_data_efficiently_crypto(self, mock_fetch):
        """Prueba 7: get_data_efficiently para cripto (con mock)"""
        from optimizador_main import get_data_efficiently
        
        # Crear mock de DataFrame
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        mock_df = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        }, index=dates)
        mock_fetch.return_value = mock_df
        
        result = get_data_efficiently("BTC/USDT", "5m", 100, "Cripto (Binance)")
        
        assert result is not None
        assert len(result) == 100
        mock_fetch.assert_called_once()
    
    def test_log_selected_parameters(self, capsys):
        """Prueba 8: log_selected_parameters imprime correctamente"""
        from optimizador_main import log_selected_parameters
        
        values = {
            "use_rsi_long": True,
            "rsi_length_min": "8",
            "rsi_length_max": "18",
            "ma_ema": True,
            "ma_sma": True,
            "ma1_min": "5",
            "ma1_max": "20",
        }
        
        log_selected_parameters(values)
        captured = capsys.readouterr()
        
        assert "PARÁMETROS SETEADOS (GUI)" in captured.out
        assert "RSI Long" in captured.out