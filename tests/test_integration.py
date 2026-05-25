"""
Pruebas de integración - Simulan ejecuciones completas
"""

import pytest
from optimizador_main import get_data_efficiently, build_config


class TestIntegration:
    """Pruebas de integración de múltiples componentes"""
    
    def test_data_download_and_backtest_flow(self):
        """Prueba 10: Flujo completo de descarga + backtest básico"""
        # Esta prueba es opcional y puede omitirse si tarda mucho
        pytest.skip("Prueba de integración completa - requiere datos reales")
    
    def test_config_build_from_values(self):
        """Prueba 11: Construcción de configuración desde valores GUI"""
        mock_values = {
            "ma_ema": True,
            "ma_sma": True,
            "ma1_min": "5",
            "ma1_max": "20",
            "ma2_min": "20",
            "ma2_max": "100",
            "rsi_length_min": "8",
            "rsi_length_max": "18",
            # ... más campos mock
        }
        
        config = build_config(mock_values)
        assert config is not None
        assert "tipos_ma" in config
        assert len(config["tipos_ma"]) >= 1