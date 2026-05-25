"""
Pruebas para el módulo de indicadores técnicos
"""

import pytest
import pandas as pd
import numpy as np
from indicators import ema, sma, rsi_tv, adx_tv


class TestIndicators:
    """Pruebas para funciones de indicadores"""
    
    @pytest.fixture
    def sample_series(self):
        """Crea una serie de precios de prueba"""
        return pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])
    
    def test_ema_calculation(self, sample_series):
        """Prueba 1: Verificar que EMA calcula correctamente"""
        result = ema(sample_series, 5)
        assert len(result) == len(sample_series)
        assert not result.isna().all()  # No todos son NaN
        assert result.iloc[-1] > 0      # El último valor es positivo
    
    def test_sma_calculation(self, sample_series):
        """Prueba 2: Verificar que SMA calcula correctamente"""
        result = sma(sample_series, 5)
        assert len(result) == len(sample_series)
        # Los primeros 4 deben ser NaN (ventana insuficiente)
        assert result.iloc[0:4].isna().all()
        # El resto debe tener valores
        assert not result.iloc[4:].isna().any()
    
    def test_rsi_tv_bounds(self, sample_series):
        """Prueba 3: Verificar que RSI está entre 0 y 100"""
        result = rsi_tv(sample_series, 5)
        # Eliminar NaN
        result_clean = result.dropna()
        assert (result_clean >= 0).all()
        assert (result_clean <= 100).all()
    
    def test_adx_tv_output_shape(self, sample_series):
        """Prueba 4: Verificar que ADX retorna 3 arrays del mismo largo"""
        high = sample_series * 1.02
        low = sample_series * 0.98
        close = sample_series
        
        adx, plus_di, minus_di = adx_tv(high, low, close, 5)
        
        assert len(adx) == len(sample_series)
        assert len(plus_di) == len(sample_series)
        assert len(minus_di) == len(sample_series)