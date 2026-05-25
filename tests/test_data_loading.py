"""
Pruebas para la carga y limpieza de datos
"""

import pytest
import pandas as pd
from optimizador_main import clean_yfinance_outliers


class TestDataLoading:
    """Pruebas para funciones de datos"""
    
    @pytest.fixture
    def df_with_outliers(self):
        """Crea un DataFrame con outliers artificiales"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        prices = [100 + i * 0.1 for i in range(100)]
        
        # Agregar un outlier (pico)
        prices[50] = 500
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        }, index=dates)
        
        return df
    
    def test_clean_yfinance_outliers_removes_extreme_values(self, df_with_outliers):
        """Prueba 8: Verificar que se eliminan outliers extremos"""
        df_clean = clean_yfinance_outliers(df_with_outliers, "TEST", "5m")
        
        # El outlier debería haber sido eliminado o corregido
        assert len(df_clean) < len(df_with_outliers)
    
    def test_clean_yfinance_outliers_preserves_normal_data(self):
        """Prueba 9: Datos normales no deben modificarse"""
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        prices = [100 + i for i in range(50)]
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        }, index=dates)
        
        original_len = len(df)
        df_clean = clean_yfinance_outliers(df, "TEST", "5m")
        
        # Datos normales no deberían modificarse
        assert len(df_clean) == original_len