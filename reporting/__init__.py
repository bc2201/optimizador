"""
Módulo de Reportes - Generación de reportes ASCII y logs
=========================================================
"""

from .ascii_reports import (
    build_ascii_table,
    generar_reporte_ascii,
    guardar_reporte_txt,
    loguear_reporte_en_console,
    generar_tabla_overfitting  # <-- AGREGAR ESTA LÍNEA
)

__all__ = [
    'build_ascii_table',
    'generar_reporte_ascii',
    'guardar_reporte_txt',
    'loguear_reporte_en_console',
    'generar_tabla_overfitting'  # <-- AGREGAR ESTA LÍNEA
]