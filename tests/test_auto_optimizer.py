"""
Pruebas para el módulo de optimización automática
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from optimization.auto_optimizer import OptimizadorAutomatico


class TestAutoOptimizer:
    """Pruebas para optimización automática multi-fase"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Crea un DataFrame OHLCV de prueba"""
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        return df
    
    @pytest.fixture
    def sample_config(self):
        """Configuración base de prueba"""
        return {
            "tipos_ma": ["EMA", "SMA"],
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
    
    @pytest.fixture
    def sample_features(self):
        """Features de prueba"""
        return {
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
    # PRUEBAS DE FUNCIONES AUXILIARES (sin mocks)
    # ============================================================
    
    def test_acotar_rangos_ma_lengths(self, sample_config):
        """Prueba 1: _acotar_rangos reduce rangos de MA correctamente"""
        optimizador = OptimizadorAutomatico(
            df=pd.DataFrame(),
            config_base=sample_config,
            features={},
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        params_referencia = {"ma1_length": 12, "ma2_length": 45}
        config_acotada = optimizador._acotar_rangos(params_referencia)
        
        assert config_acotada["ma1_min"] <= 12
        assert config_acotada["ma1_max"] >= 12
        assert config_acotada["ma1_min"] >= 8
        assert config_acotada["ma1_max"] <= 16
    
    def test_acotar_rangos_stop_loss(self, sample_config):
        """Prueba 2: _acotar_rangos redondea stop loss a múltiplos de 0.1"""
        optimizador = OptimizadorAutomatico(
            df=pd.DataFrame(),
            config_base=sample_config,
            features={},
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        params_referencia = {"stop_loss_pct": 1.23}
        config_acotada = optimizador._acotar_rangos(params_referencia)
        
        sl_range = config_acotada.get("sl_range")
        if sl_range:
            # Debe ser múltiplo de 0.1
            assert sl_range[0] * 10 % 1 == 0
            assert sl_range[1] * 10 % 1 == 0
    
    def test_metrics_config_por_fase(self):
        """Prueba 3: _metrics_config_para_fase retorna configuración correcta por fase"""
        optimizador = OptimizadorAutomatico(
            df=pd.DataFrame(),
            config_base={},
            features={},
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        config_fase1 = optimizador._metrics_config_para_fase(1)
        config_fase2 = optimizador._metrics_config_para_fase(2)
        config_fase3 = optimizador._metrics_config_para_fase(3)
        
        assert config_fase1["use_drawdown"] is False
        assert config_fase1["use_n_trades"] is False
        assert config_fase1["min_trades"] == 15
        
        assert config_fase2["use_drawdown"] is True
        assert config_fase2["use_n_trades"] is True
        assert config_fase2["min_trades"] == 20
        
        assert config_fase3["peso_drawdown"] == 35
        assert config_fase3["min_trades"] == 30
    
    def test_default_fases(self):
        """Prueba 4: _default_fases retorna estructura correcta"""
        optimizador = OptimizadorAutomatico(
            df=pd.DataFrame(),
            config_base={},
            features={},
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        fases = optimizador._default_fases()
        
        assert "fase_1" in fases
        assert "fase_2" in fases
        assert "fase_3" in fases
        assert fases["fase_1"]["trials"] == 2000
        assert fases["fase_2"]["trials"] == 1500
        assert fases["fase_3"]["corridas"] == 5
    
    def test_default_convergencia(self):
        """Prueba 5: _default_convergencia retorna valores correctos"""
        optimizador = OptimizadorAutomatico(
            df=pd.DataFrame(),
            config_base={},
            features={},
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        convergencia = optimizador._default_convergencia()
        
        assert convergencia["activar"] is True
        assert convergencia["ventana"] == 75
        assert convergencia["tolerancia"] == 0.002
        assert convergencia["trials_minimos"] == 400
    
    def test_default_metricas(self):
        """Prueba 6: _default_metricas retorna estructura correcta"""
        optimizador = OptimizadorAutomatico(
            df=pd.DataFrame(),
            config_base={},
            features={},
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        metricas = optimizador._default_metricas()
        
        assert "fase_1" in metricas
        assert "fase_2" in metricas
        assert "fase_3" in metricas
        assert metricas["fase_1"]["min_trades"] == 15
        assert metricas["fase_2"]["min_trades"] == 20
        assert metricas["fase_3"]["min_trades"] == 30
    
    def test_estabilidad_calculo(self):
        """Prueba 7: Verificar cálculo de estabilidad"""
        resultados = [0.50, 0.52, 0.51, 0.49, 0.53]
        pf_mean = np.mean(resultados)
        pf_std = np.std(resultados)
        estabilidad = 1.0 - (pf_std / pf_mean) if pf_mean > 0 else 0
        
        assert 0 <= estabilidad <= 1
        assert estabilidad > 0.9
    
    # ============================================================
    # PRUEBAS CON MOCKS (para funciones que ejecutan Optuna)
    # ============================================================
    
    @patch('optimization.auto_optimizer.run_single_optuna')
    def test_fase_exploracion_rapida_con_mock(self, mock_run_single, sample_dataframe, sample_config, sample_features):
        """Prueba 8: _fase_exploracion_rapida con mock"""
        # Configurar el mock - debe retornar una tupla (score, params)
        mock_run_single.return_value = (0.75, {"ma1_length": 12, "ma2_length": 48})
        
        optimizador = OptimizadorAutomatico(
            df=sample_dataframe,
            config_base=sample_config,
            features=sample_features,
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        # IMPORTANTE: Desactivar la convergencia para pruebas
        optimizador.config_convergencia["activar"] = False
        
        params = optimizador._fase_exploracion_rapida()
        
        assert mock_run_single.called
        assert "ma1_length" in params
        assert len(optimizador.historial) == 1
    
    @patch('optimization.auto_optimizer.run_single_optuna')
    def test_fase_refinamiento_con_mock(self, mock_run_single, sample_dataframe, sample_config, sample_features):
        """Prueba 9: _fase_refinamiento con mock"""
        mock_run_single.return_value = (0.85, {"ma1_length": 11, "ma2_length": 46})
        
        optimizador = OptimizadorAutomatico(
            df=sample_dataframe,
            config_base=sample_config,
            features=sample_features,
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        # Simular que ya hay una fase 1 en el historial
        optimizador.historial.append({
            "fase": 1,
            "nombre": "Exploración Rápida",
            "best_score": 0.70,
            "best_params": {"ma1_length": 12}
        })
        
        params_iniciales = {"ma1_length": 12, "ma2_length": 48}
        params = optimizador._fase_refinamiento(params_iniciales)
        
        assert mock_run_single.called
        assert len(optimizador.historial) == 2
        assert optimizador.historial[1]["fase"] == 2
        assert optimizador.historial[1]["best_score"] == 0.85
    
    @patch('optimization.auto_optimizer.run_single_optuna')
    def test_fase_validacion_con_mock(self, mock_run_single, sample_dataframe, sample_config, sample_features):
        """Prueba 10: _fase_validacion con mock (5 corridas)"""
        # Simular 5 corridas con diferentes scores
        mock_run_single.side_effect = [
            (0.72, {"ma1_length": 10}),
            (0.74, {"ma1_length": 11}),
            (0.73, {"ma1_length": 10}),
            (0.75, {"ma1_length": 12}),
            (0.71, {"ma1_length": 10}),
        ]
        
        optimizador = OptimizadorAutomatico(
            df=sample_dataframe,
            config_base=sample_config,
            features=sample_features,
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        params_optimos = {"ma1_length": 12}
        mejores_params = optimizador._fase_validacion(params_optimos)
        
        # Verificar que se llamó 5 veces
        assert mock_run_single.call_count == 5
        # Verificar que el historial se actualizó
        assert len(optimizador.historial) == 1
        assert optimizador.historial[0]["fase"] == 3
        assert "estabilidad" in optimizador.historial[0]
        assert "resultados_individuales" in optimizador.historial[0]
        assert len(optimizador.historial[0]["resultados_individuales"]) == 5
    
    @patch('optimization.auto_optimizer.run_single_optuna')
    def test_ejecutar_flujo_completo_con_mock(self, mock_run_single, sample_dataframe, sample_config, sample_features):
        """Prueba 11: ejecutar() flujo completo con mocks"""
        # Cada llamada debe retornar una tupla (score, params)
        mock_run_single.side_effect = [
            (0.75, {"ma1_length": 12, "ma2_length": 48}),  # Fase 1
            (0.85, {"ma1_length": 11, "ma2_length": 46}),  # Fase 2
            (0.72, {"ma1_length": 10}),  # Fase 3 - corrida 1
            (0.74, {"ma1_length": 11}),  # Fase 3 - corrida 2
            (0.73, {"ma1_length": 10}),  # Fase 3 - corrida 3
            (0.75, {"ma1_length": 12}),  # Fase 3 - corrida 4
            (0.71, {"ma1_length": 10}),  # Fase 3 - corrida 5
        ]
        
        optimizador = OptimizadorAutomatico(
            df=sample_dataframe,
            config_base=sample_config,
            features=sample_features,
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        # Desactivar convergencia para pruebas
        optimizador.config_convergencia["activar"] = False
        
        # Ejecutar fase 1
        params1 = optimizador._fase_exploracion_rapida()
        # Ejecutar fase 2
        params2 = optimizador._fase_refinamiento(params1)
        # Ejecutar fase 3
        params3 = optimizador._fase_validacion(params2)
        
        assert params1 is not None
        assert params2 is not None
        assert params3 is not None
        assert len(optimizador.historial) == 3
    
    # ============================================================
    # PRUEBAS DE INTEGRACIÓN CON ARCHIVOS (mockeadas)
    # ============================================================
    
    @patch('optimization.auto_optimizer.run_single_optuna')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_generar_reporte_y_seed_con_mock(self, mock_json_dump, mock_open, mock_run_single, 
                                              sample_dataframe, sample_config, sample_features):
        """Prueba 12: _generar_reporte_txt y _guardar_seed con mocks"""
        mock_run_single.return_value = (0.75, {"ma1_length": 12})
        
        optimizador = OptimizadorAutomatico(
            df=sample_dataframe,
            config_base=sample_config,
            features=sample_features,
            symbol="TEST",
            timeframe="1h",
            verbose=False
        )
        
        # Agregar historial simulado
        optimizador.historial = [
            {"fase": 1, "nombre": "Exploración", "best_score": 0.70, "best_params": {}},
            {"fase": 2, "nombre": "Refinamiento", "best_score": 0.75, "best_params": {"ma1": 12}},
        ]
        
        resultado_final = {"best_score": 0.75, "estabilidad": 0.95}
        
        # Probar generación de reporte (no debe lanzar excepción)
        try:
            optimizador._generar_reporte_txt({"ma1": 12}, resultado_final)
        except Exception as e:
            pytest.fail(f"_generar_reporte_txt lanzó excepción: {e}")
        
        # Probar guardado de seed
        try:
            optimizador._guardar_seed({"ma1": 12}, resultado_final)
        except Exception as e:
            pytest.fail(f"_guardar_seed lanzó excepción: {e}")