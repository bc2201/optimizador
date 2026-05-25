#!/usr/bin/env python
"""
SISTEMA DE PRUEBAS AUTOMATIZADAS
================================
Ejecuta todas las pruebas y genera reporte de resultados.
Uso: python run_tests.py
"""

import subprocess
import sys
import os
from datetime import datetime


def print_header(title):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name, status, duration, details=""):
    """Imprime resultado de prueba formateado"""
    icon = "✅" if status == "PASS" else "❌"
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    
    print(f"{icon} {test_name:<50} {color}{status:<6}{reset} ({duration:.2f}s)")
    if details and status != "PASS":
        print(f"     └─ {details}")


def run_tests():
    """Ejecuta todas las pruebas"""
    print_header("🧪 SISTEMA DE PRUEBAS - OPTIMIZADOR DE TRADING")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # 1. Pruebas de indicadores
    print_header("📊 1. Pruebas de Indicadores Técnicos")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_indicators.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    results.append(("Indicadores", result.returncode == 0, result))
    
    # 2. Pruebas de backtest
    print_header("⚙️ 2. Pruebas de Backtest")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_backtest.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    results.append(("Backtest", result.returncode == 0, result))
    
    # 3. Pruebas de datos
    print_header("💾 3. Pruebas de Carga de Datos")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_data_loading.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    results.append(("Carga de Datos", result.returncode == 0, result))
    
    # 4. Pruebas de integración
    print_header("🔗 4. Pruebas de Integración")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_integration.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    results.append(("Integración", result.returncode == 0, result))
    
    # 5. Resumen final
    print_header("📋 RESUMEN DE PRUEBAS")
    
    passed = sum(1 for _, status, _ in results if status)
    failed = len(results) - passed
    
    print(f"\n  ✅ Pruebas exitosas: {passed}")
    print(f"  ❌ Pruebas fallidas: {failed}")
    print(f"  📊 Total: {len(results)}")
    
    if failed == 0:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
    else:
        print(f"\n⚠️ {failed} prueba(s) fallaron. Revisa los detalles arriba.")
    
    print("\n" + "=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)