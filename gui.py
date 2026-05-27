"""
OPTIMIZADOR DE ESTRATEGIAS - INTERFAZ GRÁFICA (GUI)
====================================================
Autor: Trading System
Versión: 2.0

Este archivo contiene la interfaz gráfica de usuario (GUI) para el optimizador
de estrategias de trading. Permite configurar parámetros, ejecutar optimizaciones
con Optuna y visualizar resultados.

Dependencias principales:
    - PySimpleGUI: Interfaz gráfica
    - threading: Ejecución en segundo plano (no bloquea la UI)
    - optimizador: Lógica principal de backtesting y optimización
"""

import PySimpleGUI as sg
import threading
import json
import os
import webbrowser
import subprocess

# Importaciones del módulo de optimización
from optimizador_main import (
    run_optuna_with_gui,           # Función principal de optimización
    check_available_candles,       # Verifica disponibilidad de datos históricos
    progress_queue,                # Cola para actualizar barra de progreso
    generar_grafico,               # Genera gráfico HTML interactivo
    generar_preview_velas,         # Genera preview de velas (sin trades)
    calcular_trials_recomendados,  # Sugiere cantidad de trials según espacio de búsqueda
    build_config                   # Construye configuración desde GUI
)

from config import DEFAULT_RANGES

from config_auto_optimizer import DEFAULTS_AUTO


# Cambiar al directorio del script para rutas relativas
os.chdir(os.path.dirname(os.path.abspath(__file__)))



# ============================================================================
# CONFIGURACIÓN DE COLORES Y TEMAS
# ============================================================================

# Paleta de colores profesional para trading
COLORS = {
    # Colores base
    "bg_dark": "#262931",        # Fondo principal  #242424
    "bg_medium": "#444444",      # Fondo secundario (campos de ingreso de números)
    "bg_light": "#0f3460",       # Fondo de elementos
    "bg_card": "#1e2746",        # Fondo de tarjetas/secciones
    
    # Texto
    "text_primary": "#b1b1b1",   # Texto principal
    "text_secondary": "#a0a0a0", # Texto secundario
    "text_muted": "#6c6c8a",     # Texto deshabilitado
    
    # Colores de acción
    "success": "#303953",        # Verde éxito (ej: LONG) botón ejecutar
    "danger": "#47547A",         # Rojo peligro (ej: SHORT) botón detener
    "warning": "#47547A",        # Naranja advertencia; botón reset
    "info": "#1f1f1f",           # Azul información; botón abrir carpeta y ver gráfico
    "info1": "#7a7a7a",          # Titulos left panel
    
    # Colores de UI
    "border": "#2c3e66",         # Color de bordes
    "highlight": "#00bcd4",      # Resaltado
    "auto_color": "#ff6d00",     # Color especial para checkboxes "Auto"
    
    # Estados de validación
    "valid": "#1b5e20",          # Verde oscuro para campos válidos
    "invalid": "#b71c1c",        # Rojo oscuro para campos inválidos
    "empty": "#f57f17",          # Naranja para campos vacíos
    "disabled": "#2c2c3e",       # Gris para campos deshabilitados
}

# Configuración del tema de PySimpleGUI
sg.LOOK_AND_FEEL_TABLE['TradingTheme'] = {
    'BACKGROUND': COLORS["bg_dark"],
    'TEXT': COLORS["text_primary"],
    'INPUT': COLORS["bg_medium"],
    'TEXT_INPUT': COLORS["text_primary"],
    'SCROLL': COLORS["bg_light"],
    'BUTTON': ('white', COLORS["info"]),
    'PROGRESS': ('#00c853', '#00695c'),
    'BORDER': 1,
    'SLIDER_DEPTH': 0,
    'PROGRESS_DEPTH': 0,
}

# Aplicar el tema
sg.theme('TradingTheme')




# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DE MODOS DE OPTUNA
# ============================================================================
# Cada modo define una estrategia de optimización predefinida:
#   - trials: número de iteraciones
#   - serie/paralelo: ejecución secuencial o multithreading
#   - multi_run: ejecutar múltiples corridas para evitar overfitting
#   - runs: cantidad de corridas (si multi_run está activado)
# ============================================================================


MODOS_OPTUNA = {
    "Exploración Rápida": {
        "trials": 1000,
        "serie": False,
        "paralelo": True,
        "multi_run": False,
        "runs": 1,
        "descripcion": "Ideal para detectar rápidamente si la configuración general tiene potencial.",
        "print": "[MODO] Exploración Rápida: {trials} trials, paralelo, {runs} corrida(s)"
    },
    "Exploración Media": {
        "trials": 1000,
        "serie": False,
        "paralelo": True,
        "multi_run": True,
        "runs": 3,
        "descripcion": "Útil para evaluar el impacto de los filtros principales sin buscar convergencia fina.",
        "print": "[MODO] Exploración Media: {trials} trials, paralelo, {runs} corridas"
    },
    "Converger": {
        "trials": 1000,
        "serie": True,
        "paralelo": False,
        "multi_run": True,
        "runs": 3,
        "descripcion": "Optimización profunda para encontrar parámetros estables y consistentes.",
        "print": "[MODO] Converger: {trials} trials, serie, {runs} corridas"
    },
    "Validar": {
        "trials": 2000,
        "serie": True,
        "paralelo": False,
        "multi_run": True,
        "runs": 5,
        "descripcion": "Validación final para confirmar robustez y evitar falsos óptimos.",
        "print": "[MODO] Validar: {trials} trials, serie, {runs} corridas"
    },
    "Personalizado": {
        "trials": None,
        "serie": None,
        "paralelo": None,
        "multi_run": None,
        "runs": None,
        "descripcion": "Configure manualmente trials, modo serie/paralelo y multi‑run según sus necesidades.",
        "print": "[MODO] Personalizado: configure manualmente los parámetros."
    }
}


# Iconos visuales para cada modo (usados en tooltips)

MODOS_ICONOS = {
    "Exploración Rápida": "🔎",
    "Exploración Media": "🧪",
    "Converger": "🎯",
    "Validar": "🛡️",
    "Personalizado": "⚙️"
}


def generar_tooltip_modos():
    """
    Genera un texto de ayuda (tooltip) que explica cada modo de optimización.
    Se muestra al pasar el mouse sobre el selector de modos.
    """
    lineas = []
    for nombre, cfg in MODOS_OPTUNA.items():
        icono = MODOS_ICONOS.get(nombre, "•")

        if nombre == "Personalizado":
            lineas.append(f"{icono} {nombre}:\n   • {cfg['descripcion']}\n")
            continue

        modo_txt = (
            f"{icono} {nombre}:\n"
            f"   • Corre {cfg['trials']} trials "
            f"{'en serie' if cfg['serie'] else 'en paralelo'}.\n"
            f"   • {cfg['descripcion']}\n"
            f"   • Multi‑run: {'Sí' if cfg['multi_run'] else 'No'}"
        )
        if cfg["multi_run"]:
            modo_txt += f" ({cfg['runs']} corridas)"
        modo_txt += ".\n"
        lineas.append(modo_txt)

    return "\n".join(lineas)


# ============================================================================
# SECCIÓN 2: VALORES POR DEFECTO PARA CAMPOS DE RANGOS
# ============================================================================
# Estos valores se usan para auto-llenar los campos cuando el usuario activa
# un checkbox o hace clic en "Reset". Son valores de inicio razonables.
# ============================================================================

FIELD_DEFAULTS = {
    "rsi_length_min": "8",      "rsi_length_max": "18",
    "rsi_min_min": "55",        "rsi_min_max": "65",
    "rsi_max_min": "35",        "rsi_max_max": "45",
    "adx_length_min": "8",      "adx_length_max": "18",
    "adx_threshold_min": "15",  "adx_threshold_max": "25",
    "lookback_min": "2",        "lookback_max": "10",
    "validation_window_min": "5", "validation_window_max": "15",
    "htf_length_min": "10",     "htf_length_max": "50",
    "stop_loss_min": "0.3",     "stop_loss_max": "2.0",
    "velas_para_be_min": "1",   "velas_para_be_max": "10",
    "tp_long_min": "0.5",       "tp_long_max": "4.0",
    "tp_short_min": "0.5",      "tp_short_max": "4.0",
    "max_losing_streak_min": "1", "max_losing_streak_max": "3",
    "cooldown_bars_min": "10",  "cooldown_bars_max": "100",
    "max_reentries_min": "1",   "max_reentries_max": "4",
    "max_post_reentries_min": "1", "max_post_reentries_max": "3",
}


# Rangos máximos permitidos para validación visual (min, max)
# DEFAULT_RANGES movido a config.py


# Mapeo: cada checkbox controla qué campos debe mostrar/llenar
CHECKBOX_FIELDS = {
    # Checkboxes principales (ON/OFF)
    "use_rsi_long":    ["rsi_length_min", "rsi_length_max", "rsi_min_min", "rsi_min_max"],
    "use_rsi_short":   ["rsi_length_min", "rsi_length_max", "rsi_max_min", "rsi_max_max"],
    "use_adx_filter":  ["adx_length_min", "adx_length_max", "adx_threshold_min", "adx_threshold_max"],
    "enable_high_condition": ["lookback_min", "lookback_max"],
    "enable_low_condition":  ["lookback_min", "lookback_max"],
    "use_validation_window": ["validation_window_min", "validation_window_max"],
    "use_htf_filter":  ["htf_length_min", "htf_length_max"],
    "use_stop_loss":   ["stop_loss_min", "stop_loss_max"],
    "activar_stop_be": ["velas_para_be_min", "velas_para_be_max"],
    "use_take_profit_long":  ["tp_long_min", "tp_long_max"],
    "use_take_profit_short": ["tp_short_min", "tp_short_max"],
    "enable_cooldown": ["max_losing_streak_min", "max_losing_streak_max", "cooldown_bars_min", "cooldown_bars_max"],
    "enable_reentry":  ["max_reentries_min", "max_reentries_max"],
    "enable_post_crossover_entry": ["max_post_reentries_min", "max_post_reentries_max"],
    # Versiones "Auto" (mismos campos que sus equivalentes ON)
    "auto_rsi_long":        ["rsi_length_min", "rsi_length_max", "rsi_min_min", "rsi_min_max"],
    "auto_rsi_short":       ["rsi_length_min", "rsi_length_max", "rsi_max_min", "rsi_max_max"],
    "auto_adx_filter":      ["adx_length_min", "adx_length_max", "adx_threshold_min", "adx_threshold_max"],
    "auto_high_condition":  ["lookback_min", "lookback_max"],
    "auto_low_condition":   ["lookback_min", "lookback_max"],
    "auto_htf_filter":      ["htf_length_min", "htf_length_max"],
    "auto_stop_loss":       ["stop_loss_min", "stop_loss_max"],
    "auto_stop_be":         ["velas_para_be_min", "velas_para_be_max"],
    "auto_tp_long":         ["tp_long_min", "tp_long_max"],
    "auto_tp_short":        ["tp_short_min", "tp_short_max"],
    "auto_cooldown":        ["max_losing_streak_min", "max_losing_streak_max", "cooldown_bars_min", "cooldown_bars_max"],
    "auto_reentry":         ["max_reentries_min", "max_reentries_max"],
    "auto_post_crossover":  ["max_post_reentries_min", "max_post_reentries_max"],
}


# ============================================================================
# SECCIÓN 3: FUNCIONES AUXILIARES DEL GUI
# ============================================================================

def fill_defaults(window, values, checkbox_key, force=False):
    """
    Rellena los campos asociados a un checkbox con los valores por defecto.
    
    Args:
        window: Objeto ventana de PySimpleGUI
        values: Diccionario con valores actuales del GUI
        checkbox_key: Clave del checkbox que se activó
        force: Si es True, sobrescribe aunque ya tenga valor (usado en Reset y Auto)
    """
    for field in CHECKBOX_FIELDS.get(checkbox_key, []):
        if force or values.get(field, "").strip() == "":
            window[field].update(FIELD_DEFAULTS[field])


def validate_range(window, key_min, key_max, default_min, default_max, active):
    """Valida visualmente los campos de rango con la paleta de colores."""
    
    vmin = window[key_min].get()
    vmax = window[key_max].get()
    
    # Caso 1: Checkbox desactivado → gris oscuro
    if not active:
        window[key_min].update(background_color=COLORS["disabled"])
        window[key_max].update(background_color=COLORS["disabled"])
        return
    
    # Caso 2: Campos vacíos → naranja (advertencia)
    if vmin == "" or vmax == "":
        window[key_min].update(background_color=COLORS["empty"])
        window[key_max].update(background_color=COLORS["empty"])
        return
    
    # Caso 3: Validar valores numéricos
    try:
        vmin = float(vmin)
        vmax = float(vmax)
    except ValueError:
        window[key_min].update(background_color=COLORS["invalid"])
        window[key_max].update(background_color=COLORS["invalid"])
        return
    
    # Caso 4: Fuera del rango permitido → rojo
    if not (default_min <= vmin <= default_max) or not (default_min <= vmax <= default_max):
        window[key_min].update(background_color=COLORS["invalid"])
        window[key_max].update(background_color=COLORS["invalid"])
        return
    
    # Caso 5: min >= max → rojo
    if vmin >= vmax:
        window[key_min].update(background_color=COLORS["invalid"])
        window[key_max].update(background_color=COLORS["invalid"])
        return
    
    # Caso 6: Todo correcto → verde
    window[key_min].update(background_color=COLORS["valid"])
    window[key_max].update(background_color=COLORS["valid"])


# ============================================================================
# SECCIÓN 4: CONSTRUCCIÓN DE PANELES DEL GUI
# ============================================================================
# El GUI está dividido en 3 columnas:
#   - Panel Izquierdo: Métricas y parámetros de la estrategia
#   - Panel Central: Configuración de activo, modos Optuna, botones
#   - Panel Derecho: Reporte y salida de consola
# ============================================================================



# ============================================================================
# PANEL DE CONFIGURACIÓN DE OPTIMIZACIÓN AUTOMÁTICA
# ============================================================================


def make_auto_config_panel():
    """Construye el panel de configuración de optimización automática"""
    
    # Checkbox con estilo
    def cb(text, key, default=False):
        return sg.Checkbox(text, key=key, default=default, enable_events=True)
    
    # Función para crear una sección de configuración de fase (estilo manual)
        # Función para crear una sección de configuración de fase (estilo manual)
        # Función para crear una sección de configuración de fase (estilo manual exacto)
        # Función para crear una sección de configuración de fase (estilo manual compacto)
    def make_fase_config(fase_num, nombre, default_trials, default_modo, default_multi_run, default_runs):
        """Crea un frame completo para una fase con el estilo exacto del modo manual"""
        
        trials_key = f"auto_fase{fase_num}_trials"
        modo_serie_key = f"auto_fase{fase_num}_modo_serie"
        modo_paralelo_key = f"auto_fase{fase_num}_modo_paralelo"
        multi_run_key = f"auto_fase{fase_num}_multi_run"
        runs_key = f"auto_fase{fase_num}_runs"
        runs_label_key = f"auto_fase{fase_num}_runs_label"
        runs_input_key = f"auto_fase{fase_num}_runs_input"
        trials_por_corrida_key = f"auto_fase{fase_num}_trials_por_corrida"
        
        layout = []
        

        # Título de la fase
        layout.append([sg.Text(f"Fase {fase_num} - {nombre}", font=("Any", 9, "bold"), text_color="#bdbdbd")])
        

        # Fila 1: Modo de ejecución (Radio buttons)
        layout.append([
            sg.Text("Modo de ejecución:", size=(14, 1)),
            sg.Radio("Serie", group_id=f"modo_fase{fase_num}", key=modo_serie_key,
                    default=(default_modo == "Serie"), enable_events=True),
            sg.Radio("Paralelo", group_id=f"modo_fase{fase_num}", key=modo_paralelo_key,
                    default=(default_modo == "Paralelo"), enable_events=True),
        ])
        

        # Fila 2: Multi-Run + Cantidad de corridas (con visibilidad controlada)
        layout.append([
            cb("Multi-Run", multi_run_key, default=default_multi_run),
            sg.Text("Cantidad de corridas:", size=(15, 1), key=runs_label_key, visible=default_multi_run),
            sg.Input(str(default_runs), key=runs_key, size=(6, 1), visible=default_multi_run),
        ])
        

        # Fila 3: Cantidad de trials (o Trials por corrida para Fase 3)
        if fase_num == 3:
            layout.append([
                sg.Text("Trials por corrida:", size=(15, 1)),
                sg.Input(str(default_trials), key=trials_por_corrida_key, size=(10, 1)),
            ])
        else:
            layout.append([
                sg.Text("Cantidad de trials:", size=(15, 1)),
                sg.Input(str(default_trials), key=trials_key, size=(10, 1)),
            ])
        
        return sg.Frame("", layout, pad=(5, 3), border_width=1, relief=sg.RELIEF_SUNKEN)
    

    # Layout principal
    layout = [
        [sg.Text("CONFIGURACIÓN DE OPTIMIZACIÓN AUTOMÁTICA", font=("Any", 11, "bold"))],
        [sg.HorizontalSeparator()],
        
        # ========== CONVERGENCIA ==========
        [sg.Text("CONVERGENCIA (Fase 1)", font=("Any", 9, "bold"), text_color="#ff6d00")],
        [cb("Activar convergencia temprana", "auto_activar_convergencia", default=True)],
        [
            sg.Text("Ventana:", size=(10, 1)), 
            sg.Input("75", key="auto_ventana", size=(8, 1)),
            sg.Text("Tolerancia (%):", size=(14, 1)), 
            sg.Input("0.2", key="auto_tolerancia", size=(8, 1)),
        ],
        [
            sg.Text("Trials mínimos:", size=(14, 1)), 
            sg.Input("400", key="auto_trials_minimos", size=(8, 1)),
            sg.Text("Score mínimo:", size=(12, 1)), 
            sg.Input("0.85", key="auto_mejor_score_minimo", size=(8, 1)),
        ],
        [sg.Text("", size=(1, 1))],
        [sg.HorizontalSeparator()],
        
        # ========== CONFIGURACIÓN DE FASES ==========
        [sg.Text("CONFIGURACIÓN DE FASES", font=("Any", 9, "bold"), text_color="#bdbdbd")],
        
        # Fase 1
        [make_fase_config(1, "Exploración", 2000, "Paralelo", False, 1)],
        
        # Fase 2
        [make_fase_config(2, "Refinamiento", 1500, "Serie", False, 1)],
        
        # Fase 3
        [make_fase_config(3, "Validación", 800, "Serie", True, 5)],
        
        [sg.HorizontalSeparator()],
        
        # ========== MÉTRICAS POR FASE ==========
        [sg.Text("MÉTRICAS POR FASE", font=("Any", 9, "bold"), text_color="#bdbdbd")],
        
        # Fase 1 - Métricas
        [sg.Text("Fase 1 - Exploración:", font=("Any", 9, "bold"), text_color="#bdbdbd")],
        [
            sg.Text("PF:", size=(3, 1)), sg.Input("60", key="auto_fase1_pf", size=(6, 1)),
            sg.Text("WR:", size=(3, 1)), sg.Input("40", key="auto_fase1_winrate", size=(6, 1)),
            sg.Text("DD:", size=(3, 1)), sg.Input("0", key="auto_fase1_drawdown", size=(6, 1)),
            sg.Text("NT:", size=(3, 1)), sg.Input("0", key="auto_fase1_n_trades", size=(6, 1)),
            sg.Text("Min Trades:", size=(10, 1)), sg.Input("15", key="auto_fase1_min_trades", size=(6, 1)),
        ],
        [sg.Text("", size=(1, 1))],
        
        # Fase 2 - Métricas
        [sg.Text("Fase 2 - Refinamiento:", font=("Any", 9, "bold"), text_color="#bdbdbd")],
        [
            sg.Text("PF:", size=(3, 1)), sg.Input("40", key="auto_fase2_pf", size=(6, 1)),
            sg.Text("WR:", size=(3, 1)), sg.Input("30", key="auto_fase2_winrate", size=(6, 1)),
            sg.Text("DD:", size=(3, 1)), sg.Input("20", key="auto_fase2_drawdown", size=(6, 1)),
            sg.Text("NT:", size=(3, 1)), sg.Input("10", key="auto_fase2_n_trades", size=(6, 1)),
            sg.Text("Min Trades:", size=(10, 1)), sg.Input("20", key="auto_fase2_min_trades", size=(6, 1)),
        ],
        [sg.Text("", size=(1, 1))],
        
        # Fase 3 - Métricas
        [sg.Text("Fase 3 - Validación:", font=("Any", 9, "bold"), text_color="#bdbdbd")],
        [
            sg.Text("PF:", size=(3, 1)), sg.Input("35", key="auto_fase3_pf", size=(6, 1)),
            sg.Text("WR:", size=(3, 1)), sg.Input("30", key="auto_fase3_winrate", size=(6, 1)),
            sg.Text("DD:", size=(3, 1)), sg.Input("35", key="auto_fase3_drawdown", size=(6, 1)),
            sg.Text("NT:", size=(3, 1)), sg.Input("0", key="auto_fase3_n_trades", size=(6, 1)),
            sg.Text("Min Trades:", size=(10, 1)), sg.Input("30", key="auto_fase3_min_trades", size=(6, 1)),
        ],
        
        [sg.HorizontalSeparator()],
        [sg.Text("💡 Los cambios se aplican en la próxima optimización automática", 
                 font=("Any", 8), text_color="#777777")],
        [sg.Text("📌 Si activas convergencia, Fase 1 se detendrá automáticamente cuando converja", 
                 font=("Any", 8), text_color="#777777")],
    ]
    
    return sg.Column(layout, key="auto_config_panel", scrollable=True, 
                     vertical_scroll_only=True, expand_y=True)



def make_left_panel():
    """Construye el panel izquierdo con todas las métricas y parámetros."""
    
    # Función interna para crear títulos de sección
    def section_title(text):
        return [
            [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],
            [sg.Text(text, font=("Any", 9, "bold"), text_color=COLORS["info1"], pad=(0, 2))],
        ]

    # Función para crear fila de rango (label + min + flecha + max)
    def rng(label, key_min, key_max):
        return [
            sg.Text(label, size=(18, 1)),
            sg.Input(key=key_min, size=(5, 1), enable_events=True),
            sg.Text("→"),
            sg.Input(key=key_max, size=(5, 1), enable_events=True)
        ]

    # Función para crear columnas "pinneables" (pueden ocultarse/mostrarse)
    def pinned(key, rows, visible=False):
        return sg.pin(sg.Column(rows, key=key, visible=visible, pad=(0, 0)))

    # Función para checkbox simple
    def cb(text, key, default=False):
        return sg.Checkbox(text, key=key, default=default, enable_events=True)

    # Función para fila con checkbox ON/OFF + checkbox "Auto" al lado
    def cb_auto_row(label, key_on, key_auto, default_on=False):
        """Fila con checkbox ON/OFF + checkbox Auto al lado (con colores mejorados)."""
        return [
            sg.Checkbox(label, key=key_on, default=default_on, enable_events=True,
                    text_color=COLORS["text_primary"]),
            sg.Push(),
            sg.Checkbox("Auto", key=key_auto, default=False, enable_events=True,
                    text_color=COLORS["auto_color"], font=("Any", 8, "bold")),
        ]

    # Función para fila de métricas (checkbox + peso)
    def metrica_row(label, cb_key, peso_key, default_peso, default_cb=True):
        return [
            sg.Checkbox(label, key=cb_key, default=default_cb, enable_events=True),
            sg.Push(),
            sg.Text("Peso:", pad=((8, 2), 0)),
            sg.Input(default_text=str(default_peso), key=peso_key, size=(5, 1)),
            sg.Text("%"),
        ]

    # Layout completo del panel izquierdo
    layout = [
        # --- Módulo de Métricas ---
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],
        [sg.Text("MÉTRICAS DE OPTIMIZACIÓN", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        metrica_row("Profit Factor", "metric_pf", "peso_pf", 50),
        metrica_row("Win Rate", "metric_winrate", "peso_winrate", 30),
        metrica_row("Max Drawdown", "metric_drawdown", "peso_drawdown", 20),
        metrica_row("N° de Trades", "metric_n_trades", "peso_n_trades", 0, default_cb=False),
        [sg.Text("  Los pesos se normalizan automáticamente", font=("Any", 7), text_color="#777777", pad=(0, 2))],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],
        [sg.Text("Mínimo de trades:", size=(18, 1)),
         sg.Input("30", key="min_trades", size=(6, 1)),
         sg.Text("(combinaciones con menos trades se descartan)", font=("Any", 7), text_color="#777777")],

        # --- Módulo de TENDENCIA ---
        *section_title("TENDENCIA"),
        cb_auto_row("RSI Long", "use_rsi_long", "auto_rsi_long"),
        cb_auto_row("RSI Short", "use_rsi_short", "auto_rsi_short"),
        [pinned("col_rsi", [
            rng("RSI length:", "rsi_length_min", "rsi_length_max"),
            rng("RSI min (Long):", "rsi_min_min", "rsi_min_max"),
            rng("RSI max (Short):", "rsi_max_min", "rsi_max_max"),
        ])],
        cb_auto_row("ADX Filter", "use_adx_filter", "auto_adx_filter"),
        [pinned("col_adx", [
            rng("ADX length:", "adx_length_min", "adx_length_max"),
            rng("ADX umbral:", "adx_threshold_min", "adx_threshold_max"),
        ])],

        # --- Módulo de CONDICIONES DE PRECIO ---
        *section_title("CONDICIONES DE PRECIO"),
        cb_auto_row("High Condition", "enable_high_condition", "auto_high_condition", default_on=True),
        cb_auto_row("Low Condition", "enable_low_condition", "auto_low_condition", default_on=True),
        [pinned("col_hl", [rng("Lookback:", "lookback_min", "lookback_max")], visible=True)],
        [cb("Validation Window", "use_validation_window", default=True)],
        [pinned("col_valwin", [rng("Validation window:", "validation_window_min", "validation_window_max")], visible=True)],
        cb_auto_row("HTF Filter", "use_htf_filter", "auto_htf_filter"),
        [pinned("col_htf", [rng("HTF length:", "htf_length_min", "htf_length_max")])],

        # --- Módulo de GESTIÓN DE RIESGO ---
        *section_title("GESTIÓN DE RIESGO"),
        cb_auto_row("Stop Loss", "use_stop_loss", "auto_stop_loss"),
        [pinned("col_sl", [rng("Stop loss %:", "stop_loss_min", "stop_loss_max")])],
        cb_auto_row("Break Even", "activar_stop_be", "auto_stop_be"),
        [pinned("col_be", [rng("Velas para BE:", "velas_para_be_min", "velas_para_be_max")])],
        cb_auto_row("Take Profit Long", "use_take_profit_long", "auto_tp_long"),
        cb_auto_row("Take Profit Short", "use_take_profit_short", "auto_tp_short"),
        [pinned("col_tp", [
            rng("TP long %:", "tp_long_min", "tp_long_max"),
            rng("TP short %:", "tp_short_min", "tp_short_max"),
        ])],

        # --- Módulo de GESTIÓN DE OPERACIONES ---
        *section_title("GESTIÓN DE OPERACIONES"),
        cb_auto_row("Cooldown", "enable_cooldown", "auto_cooldown"),
        [pinned("col_cooldown", [
            rng("Max losing streak:", "max_losing_streak_min", "max_losing_streak_max"),
            rng("Cooldown bars:", "cooldown_bars_min", "cooldown_bars_max"),
        ])],
        cb_auto_row("Reentry", "enable_reentry", "auto_reentry"),
        [pinned("col_reentry", [rng("Max reentries:", "max_reentries_min", "max_reentries_max")])],
        cb_auto_row("Post Crossover Entry", "enable_post_crossover_entry", "auto_post_crossover"),
        [pinned("col_post_reentry", [rng("Max post reentries:", "max_post_reentries_min", "max_post_reentries_max")])],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],
    ]

    return sg.Column(
        layout,
        key="left_scroll",
        pad=(4, 4),
        expand_x=True,
        scrollable=True,
        vertical_scroll_only=True,
        expand_y=True,
    )



def make_center_panel():
    """Construye el panel central con configuración de activo y controles principales."""
    return sg.Column([
        # --- Fuente de datos (Cripto vs Acciones) ---
        [sg.Text("FUENTE DE DATOS", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [
            sg.Radio("Cripto (Binance)", group_id="fuente_grupo", key="data_source_crypto", 
                     default=True, font=("Any", 10)),
            sg.Radio("Acciones (Yahoo Finance)", group_id="fuente_grupo", key="data_source_yf", 
                     font=("Any", 10))
        ],
        
        # Después de los radios de fuente de datos
        [sg.Text("● Sistema listo", key="status_indicator", text_color=COLORS["success"], 
                font=("Any", 8), pad=((0, 0), (5, 0)))],

        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # --- Configuración del Activo ---
        [sg.Text("CONFIGURACIÓN DE ACTIVO", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [sg.Text("Símbolo (ej: SOL/USDT):"), sg.Input(key="symbol", size=(12, 1))],
        [sg.Text("Timeframe:"), sg.Combo(["1m", "5m", "15m", "1h", "4h", "1d"], key="timeframe", default_value="5m")],
        [sg.Text("Cantidad de velas:"), sg.Input(key="candles", size=(10, 1))],
        [
            sg.Button("Verificar velas", key="check_candles", button_color=("white", "#254166")),
            sg.Button("📊 Preview velas", key="preview_velas", button_color=("white", "#254166"))
        ],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # --- Dirección de Trades ---
        [sg.Text("DIRECCIÓN DE TRADES", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [
            sg.Checkbox("Habilitar Longs", key="enable_long_trades", default=True),
            sg.Checkbox("Habilitar Shorts", key="enable_short_trades", default=True)
        ],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # --- Tipos de Medias Móviles ---
        [sg.Text("MEDIAS MÓVILES", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [
            sg.Checkbox("EMA", key="ma_ema", default=True),
            sg.Checkbox("SMA", key="ma_sma", default=True),
            sg.Checkbox("WMA", key="ma_wma"),
            sg.Checkbox("HMA", key="ma_hma"),
            sg.Checkbox("DEMA", key="ma_dema")
        ],
        [sg.Text("MA1 rango:"), sg.Input(key="ma1_min", size=(5, 1)), sg.Text("→"), sg.Input(key="ma1_max", size=(5, 1))],
        [sg.Text("MA2 rango:"), sg.Input(key="ma2_min", size=(5, 1)), sg.Text("→"), sg.Input(key="ma2_max", size=(5, 1))],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # --- Modo Optuna ---
        [sg.Text("MODO OPTUNA", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [sg.Combo(
            ["Exploración Rápida", "Exploración Media", "Converger", "Validar", "Personalizado"],
            key="modo_opt", default_value="Exploración Rápida", enable_events=True,
            size=(20, 1), tooltip=generar_tooltip_modos()
        )],
        [sg.Text("MODO DE EJECUCIÓN", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [sg.Radio("Serie", "MODO", key="modo_serie", default=True), sg.Radio("Paralelo", "MODO", key="modo_paralelo")],
        [sg.Checkbox("Multi‑Run", key="multi_run", default=False)],
        [
            sg.Text("Cantidad de corridas:"), sg.Input("1", key="multi_runs_count", size=(6, 1)),
            sg.Text("Cantidad de trials:"), sg.Input(key="trials", size=(10, 1)),
            sg.Button("💡 Calcular", key="calcular_trials", button_color=("white", "#254166"),
                      tooltip="Analiza el espacio de búsqueda y sugiere un rango de trials")
        ],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # --- Prevención de Overfitting ---
        [sg.Text("PREVENCIÓN DE OVERFITTING", font=("Any", 9, "bold"), text_color="#bdbdbd", pad=(0, 2))],
        [sg.Checkbox("Activar validación IS/OOS", key="use_oos_validation", default=False, enable_events=True)],
        [sg.Text("% Datos Entrenamiento:"), sg.Input("70", key="oos_train_pct", size=(5, 1))],
        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # === 6) BOTONERA ===
        # ============================================================
        # BOTONERA PRINCIPAL
        # ============================================================
        [
            sg.Button("📊 Optimización Manual", button_color=(COLORS["text_primary"], COLORS["success"]), 
                    key="run", size=(20, 1)),
            sg.Button("🚀 Optimización Automática", button_color=(COLORS["text_primary"], "#9b59b6"), 
                    key="run_auto", size=(22, 1)),
            sg.Button("⏹ Detener", key="stop", button_color=(COLORS["text_primary"], COLORS["danger"]), 
                    disabled=True, size=(14, 1)),
            sg.Button("Reset", key="reset", button_color=(COLORS["text_primary"], COLORS["warning"]), 
                    size=(12, 1)),
        ],

        [sg.Text("─" * 60, font=("Any", 8), text_color="#383838")],

        # ============================================================
        # BOTONERA SECUNDARIA
        # ============================================================
        [
            sg.Button("Abrir carpeta", key="open_reports", 
                    button_color=(COLORS["text_primary"], COLORS["info"]), size=(14, 1)),
            sg.Button("📈 Ver gráfico", key="ver_grafico", 
                    button_color=(COLORS["text_primary"], COLORS["info"]), disabled=True, size=(14, 1)),
            sg.Button("Guardar preset", key="save_preset", 
                    button_color=(COLORS["text_primary"], COLORS["bg_light"]), size=(14, 1)),
            sg.Button("Cargar preset", key="load_preset", 
                    button_color=(COLORS["text_primary"], COLORS["bg_light"]), size=(14, 1)),
            sg.Button("Cargar Semilla", key="load_semilla", 
                    button_color=(COLORS["text_primary"], COLORS["bg_light"]), size=(14, 1)),
        ],


    ], vertical_alignment="top", expand_x=True, expand_y=True)



def make_right_panel():
    return sg.Column([
        [sg.Text("📋 Reporte", font=("Any", 12, "bold"), text_color=COLORS["info"])],
        [sg.Text("📊 Progreso Optuna:", text_color=COLORS["text_secondary"])],
        [sg.ProgressBar(1.0, orientation='h', size=(40, 20), key='progress', 
                       bar_color=(COLORS["success"], COLORS["bg_light"]))],
        [sg.Output(size=(70, 25), key="output", expand_x=True, expand_y=True,
                  background_color=COLORS["bg_dark"], text_color=COLORS["text_primary"])]
    ], vertical_alignment="top", expand_x=True, expand_y=True)


# ============================================================================
# SECCIÓN 5: VARIABLES GLOBALES Y FUNCIONES DE HILO
# ============================================================================

_stop_event = threading.Event()      # Evento para cancelar la optimización
_ultimo_grafico_path = None          # Ruta del último gráfico generado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(BASE_DIR, "presets")
os.makedirs(PRESET_DIR, exist_ok=True)


def optimization_thread_wrapper(window, values, stop_event):
    """
    Ejecuta la optimización en un hilo separado para no bloquear la interfaz.
    Al finalizar, envía eventos al GUI para actualizar el gráfico y el estado.
    """
    try:
        html_path = run_optuna_with_gui(values, stop_event)
        window.write_event_value("-GRAFICO_LISTO-", html_path)
    except Exception as e:
        print(f"\n[THREAD ERROR] {e}")
        window.write_event_value("-GRAFICO_LISTO-", None)
    finally:
        window.write_event_value("-OPT_FINISHED-", None)



def optimization_thread_wrapper_auto(window, values, stop_event):
    """Ejecuta la optimización automática multi-fase en hilo separado."""
    try:
        from optimization import ejecutar_optimizacion_automatica
        from optimizador_main import get_data_efficiently, build_config, FEATURES, CONSTANTS
        from config_auto_optimizer import CONFIG_FASES, CONFIG_CONVERGENCIA, CONFIG_METRICAS
        
        symbol = values["symbol"].upper().strip()
        timeframe = values["timeframe"]
        total_candles = int(values["candles"])
        
        # ============================================================
        # CONSTRUIR CONFIGURACIÓN DESDE EL GUI
        # ============================================================
        
        def get_modo(fase_num, values):
            if values.get(f"auto_fase{fase_num}_modo_serie"):
                return "serie"
            elif values.get(f"auto_fase{fase_num}_modo_paralelo"):
                return "paralelo"
            return "serie"

        config_fases = {
            "fase_1": {
                "trials": int(values.get("auto_fase1_trials", 2000)),
                "modo": get_modo(1, values),
                "multi_run": values.get("auto_fase1_multi_run", False),
                "runs": int(values.get("auto_fase1_runs", 1))
            },
            "fase_2": {
                "trials": int(values.get("auto_fase2_trials", 1500)),
                "modo": get_modo(2, values),
                "multi_run": values.get("auto_fase2_multi_run", False),
                "runs": int(values.get("auto_fase2_runs", 1))
            },
            "fase_3": {
                "trials_por_corrida": int(values.get("auto_fase3_trials_x_corrida", 800)),
                "modo": get_modo(3, values),
                "multi_run": values.get("auto_fase3_multi_run", True),
                "corridas": int(values.get("auto_fase3_runs", 5))
            }
        }
        
        # Configuración de Convergencia
        config_convergencia = {
            "activar": values.get("auto_activar_convergencia", True),
            "ventana": int(values.get("auto_ventana", 75)),
            "tolerancia": float(values.get("auto_tolerancia", 0.002)),
            "trials_minimos": int(values.get("auto_trials_minimos", 400)),
            "mejor_score_minimo": float(values.get("auto_mejor_score_minimo", 0.85))
        }
        
        # Configuración de Métricas por Fase
        def parse_float(val, default):
            try:
                return float(val) if val else default
            except:
                return default
        
        config_metricas = {
            "fase_1": {
                "use_pf": values.get("auto_fase1_use_pf", True),
                "peso_pf": parse_float(values.get("auto_fase1_pf"), 60.0),
                "use_winrate": values.get("auto_fase1_use_winrate", True),
                "peso_winrate": parse_float(values.get("auto_fase1_winrate"), 40.0),
                "use_drawdown": values.get("auto_fase1_use_drawdown", False),
                "peso_drawdown": parse_float(values.get("auto_fase1_drawdown"), 0.0),
                "use_n_trades": values.get("auto_fase1_use_n_trades", False),
                "peso_n_trades": parse_float(values.get("auto_fase1_n_trades"), 0.0),
                "min_trades": int(values.get("auto_fase1_min_trades", 15))
            },
            "fase_2": {
                "use_pf": values.get("auto_fase2_use_pf", True),
                "peso_pf": parse_float(values.get("auto_fase2_pf"), 40.0),
                "use_winrate": values.get("auto_fase2_use_winrate", True),
                "peso_winrate": parse_float(values.get("auto_fase2_winrate"), 30.0),
                "use_drawdown": values.get("auto_fase2_use_drawdown", True),
                "peso_drawdown": parse_float(values.get("auto_fase2_drawdown"), 20.0),
                "use_n_trades": values.get("auto_fase2_use_n_trades", True),
                "peso_n_trades": parse_float(values.get("auto_fase2_n_trades"), 10.0),
                "min_trades": int(values.get("auto_fase2_min_trades", 20))
            },
            "fase_3": {
                "use_pf": values.get("auto_fase3_use_pf", True),
                "peso_pf": parse_float(values.get("auto_fase3_pf"), 35.0),
                "use_winrate": values.get("auto_fase3_use_winrate", True),
                "peso_winrate": parse_float(values.get("auto_fase3_winrate"), 30.0),
                "use_drawdown": values.get("auto_fase3_use_drawdown", True),
                "peso_drawdown": parse_float(values.get("auto_fase3_drawdown"), 35.0),
                "use_n_trades": values.get("auto_fase3_use_n_trades", False),
                "peso_n_trades": parse_float(values.get("auto_fase3_n_trades"), 0.0),
                "min_trades": int(values.get("auto_fase3_min_trades", 30))
            }
        }
        
        print("\n[INFO] Descargando datos para optimización automática...")
        
        # Descargar datos
        df_full = get_data_efficiently(symbol, timeframe, total_candles, values["data_source"])
        if df_full is None or len(df_full) < 50:
            print("\n[ERROR] No se pudieron obtener suficientes velas.")
            window.write_event_value("-OPT_FINISHED-", None)
            return
        
        df_full = df_full.dropna().copy()
        print(f"[INFO] Datos descargados: {len(df_full)} velas")
        
        # Construir configuración base
        config = build_config(values)
        if config is None:
            print("ERROR: Debes seleccionar al menos un tipo de MA.")
            window.write_event_value("-OPT_FINISHED-", None)
            return
        
        # Construir features
        features = {key: values[key] if key in values else FEATURES[key] for key in FEATURES}
        
        # Mapear Auto features
        AUTO_MAP = {
            "use_rsi_long": "auto_rsi_long", "use_rsi_short": "auto_rsi_short",
            "use_adx_filter": "auto_adx_filter", "enable_high_condition": "auto_high_condition",
            "enable_low_condition": "auto_low_condition", "use_htf_filter": "auto_htf_filter",
            "use_stop_loss": "auto_stop_loss", "activar_stop_be": "auto_stop_be",
            "use_take_profit_long": "auto_tp_long", "use_take_profit_short": "auto_tp_short",
            "enable_cooldown": "auto_cooldown", "enable_reentry": "auto_reentry",
            "enable_post_crossover_entry": "auto_post_crossover",
        }
        for feat_key, auto_key in AUTO_MAP.items():
            if values.get(auto_key, False):
                features[feat_key] = "auto"
        
        # Mostrar configuración utilizada
        print("\n" + "="*60)
        print("⚙️ CONFIGURACIÓN DE OPTIMIZACIÓN AUTOMÁTICA")
        print("="*60)
        print(f"\n📊 Convergencia: {'Activada' if config_convergencia['activar'] else 'Desactivada'}")
        if config_convergencia['activar']:
            print(f"   • Ventana: {config_convergencia['ventana']} trials")
            print(f"   • Tolerancia: {config_convergencia['tolerancia']*100:.1f}%")
            print(f"   • Trials mínimos: {config_convergencia['trials_minimos']}")
        print(f"\n📈 Fase 1: {config_fases['fase_1']['trials']} trials ({config_fases['fase_1']['modo']})")
        print(f"📈 Fase 2: {config_fases['fase_2']['trials']} trials ({config_fases['fase_2']['modo']})")
        print(f"📈 Fase 3: {config_fases['fase_3']['corridas']} corridas de {config_fases['fase_3']['trials_por_corrida']} trials")
        print("="*60 + "\n")
        
        # Ejecutar optimización automática con la configuración del GUI
        resultado = ejecutar_optimizacion_automatica(
            df=df_full,
            config_base=config,
            features=features,
            symbol=symbol,
            timeframe=timeframe,
            config_fases=config_fases,
            config_convergencia=config_convergencia,
            config_metricas=config_metricas,
            verbose=True
        )
        
        print(f"\n[ÉXITO] Optimización automática completada")
        print(f"  Mejor score: {resultado['best_score']:.4f}")
        if 'estabilidad' in resultado:
            print(f"  Estabilidad: {resultado['estabilidad']:.1%}")
        
        window.write_event_value("-OPT_FINISHED-", None)
        
    except Exception as e:
        print(f"\n[THREAD ERROR] {e}")
        import traceback
        traceback.print_exc()
        window.write_event_value("-OPT_FINISHED-", None)



def save_preset(values):
    """Guarda la configuración actual del GUI en un archivo JSON."""
    preset_path = sg.popup_get_file(
        "Guardar preset como...", save_as=True, default_extension=".json",
        initial_folder=PRESET_DIR, file_types=(("JSON Files", "*.json"),)
    )
    if not preset_path:
        return

    # Claves que no tienen sentido guardar (son dinámicas o de la UI)
    excluded = {"output", "progress", "-GRAFICO_LISTO-"}
    data = {k: v for k, v in values.items() if k not in excluded}

    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    sg.popup("Preset guardado correctamente.", title="Éxito")



def load_preset(window):
    """Carga una configuración desde un archivo JSON y actualiza el GUI."""
    preset_path = sg.popup_get_file(
        "Seleccionar preset...", initial_folder=PRESET_DIR,
        file_types=(("JSON Files", "*.json"),)
    )
    if not preset_path:
        return

    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, val in data.items():
        if key in window.key_dict:
            window[key].update(val)

    window.write_event_value("preset_loaded", data)
    sg.popup("Preset cargado correctamente.", title="Éxito")



def launch_gui():
    """Construye y retorna la ventana principal del GUI con pestañas."""
    left_panel = make_left_panel()
    center_panel = make_center_panel()
    right_panel = make_right_panel()
    auto_config_panel = make_auto_config_panel()
    
    # ============================================================
    # CORRECCIÓN: Envolver los paneles en listas para las pestañas
    # ============================================================
    tab_principal = sg.Tab("Principal", [[center_panel]], key="tab_principal")
    tab_auto_config = sg.Tab("Configuración Auto", [[auto_config_panel]], key="tab_auto_config")
    
    tab_group = sg.TabGroup([[tab_principal, tab_auto_config]], 
                            expand_x=True, expand_y=True, key="tab_group")
    
    layout = [[
        sg.Column([[left_panel]], expand_x=True, expand_y=True, pad=(0, 0)),
        sg.Column([[tab_group]], expand_x=True, expand_y=True, pad=(0, 0)),
        sg.Column([[right_panel]], expand_x=True, expand_y=True, pad=(0, 0))
    ]]
    
    window = sg.Window(
        "Optimizador de Estrategias — GUI Profesional",
        layout, finalize=True, resizable=True, location=(0, 0)
    )
    window.maximize()
    
    # Precargar valores por defecto para checkboxes que arrancan activos
    _initial_values = window.read(timeout=0)[1]
    for cb_key in ["enable_high_condition", "enable_low_condition", "use_validation_window"]:
        fill_defaults(window, _initial_values, cb_key)
    
    # ============================================================
    # CARGAR VALORES POR DEFECTO DE OPTIMIZACIÓN AUTOMÁTICA
    # ============================================================
    for key, val in DEFAULTS_AUTO.items():
        if key in window.key_dict:
            window[key].update(val)
    
    return window



# ============================================================================
# SECCIÓN 6: BUCLE PRINCIPAL DEL GUI (EVENT LOOP)
# ============================================================================


def gui_main():
    """Bucle principal que procesa eventos del GUI hasta que el usuario cierra."""
    window = launch_gui()

    # Configuración inicial del modo Optuna
    initial_mode = "Exploración Rápida"
    window["modo_opt"].update(initial_mode)
    window.write_event_value("modo_opt", initial_mode)

    while True:
        event, values = window.read(timeout=50)

        # Reconstruir variable virtual 'data_source' para el resto del script
        if values:
            values["data_source"] = "Cripto (Binance)" if values.get("data_source_crypto") else "Acciones (Yahoo Finance)"

        # --- Evento: Cerrar ventana ---
        if event == sg.WINDOW_CLOSED:
            break

        # --- Evento: Actualizar barra de progreso desde la cola ---
        try:
            while True:
                current = progress_queue.get_nowait()
                window['progress'].update(current_count=current)
        except:
            pass

        # --- Evento: Fin de la optimización ---
        if event == "-OPT_FINISHED-":
            if _stop_event.is_set():
                window["status_indicator"].update("⏹️ Cancelado", text_color=COLORS["warning"])
            else:
                window["status_indicator"].update("✅ Optimización completada", text_color=COLORS["success"])
            
            window['progress'].update(current_count=1.0)
            window["run"].update(disabled=False, button_color=(COLORS["text_primary"], COLORS["success"]))
            window["run_auto"].update(disabled=False, button_color=(COLORS["text_primary"], "#9b59b6"))
            window["stop"].update(disabled=True, button_color=(COLORS["text_primary"], COLORS["danger"]))
            if _stop_event.is_set():
                print("\n[INFO] Optimización cancelada por el usuario.\n")
                sg.popup("Optimización cancelada.", title="Detenido")
            else:
                sg.popup("Optimización completada con éxito.", title="Finalizado")

        # --- Evento: Gráfico listo (abrir automáticamente) ---
        if event == "-GRAFICO_LISTO-":
            global _ultimo_grafico_path
            _ultimo_grafico_path = values["-GRAFICO_LISTO-"]
            if _ultimo_grafico_path:
                window["ver_grafico"].update(disabled=False, button_color=("white", "#5a5a8a"))
                webbrowser.open(f"file:///{_ultimo_grafico_path}")

        # --- Evento: Botón "Ver gráfico" ---
        if event == "ver_grafico":
            if _ultimo_grafico_path:
                webbrowser.open(f"file:///{_ultimo_grafico_path}")

        # --- Evento: Botón "Detener" ---
        if event == "stop":
            _stop_event.set()
            # <--- NUEVO: Actualizar indicador de estado ---
            window["status_indicator"].update("⏹️ Deteniendo...", text_color=COLORS["warning"])
            print("\n[INFO] Señal de cancelación enviada. Esperando que el trial actual termine...\n")
            window["stop"].update(disabled=True, button_color=("white", "gray"))

        # --- Evento: Selector de Modo Optuna ---
        if event == "modo_opt":
            modo = values["modo_opt"]
            cfg = MODOS_OPTUNA.get(modo)

            if cfg is None:
                print("[ERROR] Modo no encontrado:", modo)
                continue

            if cfg["trials"] is not None:
                window["trials"].update(str(cfg["trials"]))
            if cfg["paralelo"] is not None:
                window["modo_paralelo"].update(cfg["paralelo"])
            if cfg["serie"] is not None:
                window["modo_serie"].update(cfg["serie"])
            if cfg["multi_run"] is not None:
                window["multi_run"].update(cfg["multi_run"])
            if cfg["runs"] is not None:
                window["multi_runs_count"].update(str(cfg["runs"]))

            print(cfg["print"].format(trials=cfg["trials"], runs=cfg["runs"]))

        # --- Eventos: Actualizar visibilidad y Auto-rellenar (dentro de if event in values) ---
        if event in values:

            def is_on(key_on, key_auto):
                return values.get(key_on, False) or values.get(key_auto, False)

            # Deshabilitar checkbox ON/OFF si Auto está tildado
            for key_on, key_auto in [
                ("use_rsi_long", "auto_rsi_long"),
                ("use_rsi_short", "auto_rsi_short"),
                ("use_adx_filter", "auto_adx_filter"),
                ("enable_high_condition", "auto_high_condition"),
                ("enable_low_condition", "auto_low_condition"),
                ("use_htf_filter", "auto_htf_filter"),
                ("use_stop_loss", "auto_stop_loss"),
                ("activar_stop_be", "auto_stop_be"),
                ("use_take_profit_long", "auto_tp_long"),
                ("use_take_profit_short", "auto_tp_short"),
                ("enable_cooldown", "auto_cooldown"),
                ("enable_reentry", "auto_reentry"),
                ("enable_post_crossover_entry", "auto_post_crossover"),
            ]:
                window[key_on].update(disabled=values.get(key_auto, False))

            # Visibilidad de bloques según checkboxes activos
            window["col_rsi"].update(
                visible=is_on("use_rsi_long", "auto_rsi_long") or is_on("use_rsi_short", "auto_rsi_short")
            )
            window["col_adx"].update(visible=is_on("use_adx_filter", "auto_adx_filter"))
            window["col_hl"].update(
                visible=is_on("enable_high_condition", "auto_high_condition") or
                        is_on("enable_low_condition", "auto_low_condition")
            )
            window["col_valwin"].update(visible=values["use_validation_window"])
            window["col_htf"].update(visible=is_on("use_htf_filter", "auto_htf_filter"))
            window["col_sl"].update(visible=is_on("use_stop_loss", "auto_stop_loss"))
            window["col_be"].update(visible=is_on("activar_stop_be", "auto_stop_be"))
            window["col_tp"].update(
                visible=is_on("use_take_profit_long", "auto_tp_long") or
                        is_on("use_take_profit_short", "auto_tp_short")
            )
            window["col_cooldown"].update(visible=is_on("enable_cooldown", "auto_cooldown"))
            window["col_reentry"].update(visible=is_on("enable_reentry", "auto_reentry"))
            window["col_post_reentry"].update(visible=is_on("enable_post_crossover_entry", "auto_post_crossover"))

            # Auto-rellenar campos según checkbox activado
            is_auto_event = event in CHECKBOX_FIELDS and event.startswith("auto_")
            is_on_event = event in CHECKBOX_FIELDS and not event.startswith("auto_")
            if is_auto_event and values.get(event) is True:
                fill_defaults(window, values, event, force=True)
            elif is_on_event and values.get(event) is True:
                fill_defaults(window, values, event, force=False)

            # Validación visual de rangos (todos los campos)
            validate_range(window, "rsi_length_min", "rsi_length_max", *DEFAULT_RANGES["rsi_length"],
                          values["use_rsi_long"] or values["use_rsi_short"])
            validate_range(window, "rsi_min_min", "rsi_min_max", *DEFAULT_RANGES["rsi_min"], values["use_rsi_long"])
            validate_range(window, "rsi_max_min", "rsi_max_max", *DEFAULT_RANGES["rsi_max"], values["use_rsi_short"])
            validate_range(window, "adx_length_min", "adx_length_max", *DEFAULT_RANGES["adx_length"], values["use_adx_filter"])
            validate_range(window, "adx_threshold_min", "adx_threshold_max", *DEFAULT_RANGES["adx_threshold"], values["use_adx_filter"])
            validate_range(window, "lookback_min", "lookback_max", *DEFAULT_RANGES["lookback"],
                          values["enable_high_condition"] or values["enable_low_condition"])
            validate_range(window, "validation_window_min", "validation_window_max", *DEFAULT_RANGES["validation_window"],
                          values["use_validation_window"])
            validate_range(window, "htf_length_min", "htf_length_max", *DEFAULT_RANGES["htf_length"], values["use_htf_filter"])
            validate_range(window, "stop_loss_min", "stop_loss_max", *DEFAULT_RANGES["stop_loss"], values["use_stop_loss"])
            validate_range(window, "velas_para_be_min", "velas_para_be_max", *DEFAULT_RANGES["be"], values["activar_stop_be"])
            validate_range(window, "tp_long_min", "tp_long_max", *DEFAULT_RANGES["tp_long"], values["use_take_profit_long"])
            validate_range(window, "tp_short_min", "tp_short_max", *DEFAULT_RANGES["tp_short"], values["use_take_profit_short"])
            validate_range(window, "max_losing_streak_min", "max_losing_streak_max", *DEFAULT_RANGES["mls"], values["enable_cooldown"])
            validate_range(window, "cooldown_bars_min", "cooldown_bars_max", *DEFAULT_RANGES["cooldown"], values["enable_cooldown"])
            validate_range(window, "max_reentries_min", "max_reentries_max", *DEFAULT_RANGES["re"], values["enable_reentry"])
            validate_range(window, "max_post_reentries_min", "max_post_reentries_max", *DEFAULT_RANGES["postre"],
                          values["enable_post_crossover_entry"])

            # Rellenar campos vacíos al activar un checkbox
            if event in CHECKBOX_FIELDS and values.get(event) is True:
                fill_defaults(window, values, event)

            # Caso especial: preset cargado (recalcular visibilidad)
            if event == "preset_loaded":
                vals = window.read(timeout=0)[1]
                window["col_rsi"].update(
                    visible=vals["use_rsi_long"] or vals["use_rsi_short"] or
                            vals.get("auto_rsi_long", False) or vals.get("auto_rsi_short", False)
                )
                window["col_adx"].update(visible=vals["use_adx_filter"] or vals.get("auto_adx_filter", False))
                window["col_hl"].update(
                    visible=vals["enable_high_condition"] or vals["enable_low_condition"] or
                            vals.get("auto_high_condition", False) or vals.get("auto_low_condition", False)
                )
                window["col_valwin"].update(visible=vals["use_validation_window"])
                window["col_htf"].update(visible=vals["use_htf_filter"] or vals.get("auto_htf_filter", False))
                window["col_sl"].update(visible=vals["use_stop_loss"] or vals.get("auto_stop_loss", False))
                window["col_be"].update(visible=vals["activar_stop_be"] or vals.get("auto_stop_be", False))
                window["col_tp"].update(
                    visible=vals["use_take_profit_long"] or vals["use_take_profit_short"] or
                            vals.get("auto_tp_long", False) or vals.get("auto_tp_short", False)
                )
                window["col_cooldown"].update(visible=vals["enable_cooldown"] or vals.get("auto_cooldown", False))
                window["col_reentry"].update(visible=vals["enable_reentry"] or vals.get("auto_reentry", False))
                window["col_post_reentry"].update(visible=vals["enable_post_crossover_entry"] or vals.get("auto_post_crossover", False))

        

        # ============================================================
        # MANEJAR VISIBILIDAD DINÁMICA DE MULTI-RUN (Configuración Auto)
        # ============================================================
        if event in ["auto_fase1_multi_run", "auto_fase2_multi_run", "auto_fase3_multi_run"]:
            # Determinar qué fase se modificó
            if event == "auto_fase1_multi_run":
                fase = 1
            elif event == "auto_fase2_multi_run":
                fase = 2
            else:
                fase = 3
            
            visible = values[event]
            # Mostrar/ocultar label e input de corridas
            try:
                window[f"auto_fase{fase}_runs_label"].update(visible=visible)
                window[f"auto_fase{fase}_runs"].update(visible=visible)  # ← usar runs_key, no runs_input
            except Exception as e:
                print(f"Error al actualizar visibilidad: {e}")
        
        
        
        # --- Evento: Validación de timeframe para acciones (cambio en fuente o timeframe) ---
        if event in ["data_source_crypto", "data_source_yf", "timeframe"]:
            if values.get("data_source_yf") and values["timeframe"] != "1d":
                respuesta = sg.popup_yes_no(
                    f"⚠️ ADVERTENCIA: Timeframe no recomendado\n\n"
                    f"Seleccionaste Acciones con timeframe {values['timeframe']}.\n\n"
                    f"Yahoo Finance SOLO es confiable para timeframe DIARIO (1d).\n"
                    f"Para timeframes intradiarios (1m, 5m, 15m, 1h, 4h):\n"
                    f"  • Usa Cripto (Binance) - datos confiables\n\n"
                    f"¿Deseas cambiar a timeframe 1d?",
                    title="⚠️ Datos potencialmente corruptos",
                    keep_on_top=True
                )
                if respuesta == "Yes":
                    window["timeframe"].update("1d")

        # --- Evento: Botón RESET ---
        if event == "reset":
            # Limpiar booleanos y textos
            for key in values:
                if key in ["data_source", "output", "progress"]:
                    continue
                try:
                    if key not in window.key_dict:
                        continue
                    if isinstance(values[key], bool):
                        window[key].update(False)
                    elif isinstance(values[key], str):
                        window[key].update("")
                except:
                    pass

            # Resetear radio buttons a Cripto
            window["data_source_crypto"].update(True)
            window["data_source_yf"].update(False)

            # Resetear checkboxes del panel izquierdo
            window["enable_high_condition"].update(True)
            window["enable_low_condition"].update(True)
            window["use_validation_window"].update(True)

            # Resetear Auto checkboxes
            auto_keys = [
                "auto_rsi_long", "auto_rsi_short", "auto_adx_filter",
                "auto_high_condition", "auto_low_condition", "auto_htf_filter",
                "auto_stop_loss", "auto_stop_be", "auto_tp_long", "auto_tp_short",
                "auto_cooldown", "auto_reentry", "auto_post_crossover"
            ]
            for key in auto_keys:
                if key in window.key_dict:
                    window[key].update(False)

            # Resetear métricas
            window["metric_pf"].update(True)
            window["metric_winrate"].update(True)
            window["metric_drawdown"].update(True)
            window["metric_n_trades"].update(False)
            window["peso_pf"].update("50")
            window["peso_winrate"].update("30")
            window["peso_drawdown"].update("20")
            window["peso_n_trades"].update("0")
            window["min_trades"].update("30")

            # Resetear Modo Optuna a Exploración Rápida
            window["modo_opt"].update("Exploración Rápida")
            cfg_default = MODOS_OPTUNA["Exploración Rápida"]
            window["trials"].update(str(cfg_default["trials"]))
            window["modo_serie"].update(cfg_default["serie"])
            window["modo_paralelo"].update(cfg_default["paralelo"])
            window["multi_run"].update(cfg_default["multi_run"])
            window["multi_runs_count"].update(str(cfg_default["runs"]))

            # Resetear overfitting
            window["use_oos_validation"].update(False)
            window["oos_train_pct"].update("70")

            # Limpiar campos del activo
            window["symbol"].update("")
            window["timeframe"].update("5m")
            window["candles"].update("")

            # Resetear dirección de trades
            window["enable_long_trades"].update(True)
            window["enable_short_trades"].update(True)

            # Resetear tipos de medias móviles
            window["ma_ema"].update(True)
            window["ma_sma"].update(True)
            window["ma_wma"].update(False)
            window["ma_hma"].update(False)
            window["ma_dema"].update(False)
            window["ma1_min"].update("")
            window["ma1_max"].update("")
            window["ma2_min"].update("")
            window["ma2_max"].update("")

            # Limpiar todos los campos de rangos
            range_keys = [
                "rsi_length_min", "rsi_length_max", "rsi_min_min", "rsi_min_max",
                "rsi_max_min", "rsi_max_max", "adx_length_min", "adx_length_max",
                "adx_threshold_min", "adx_threshold_max", "lookback_min", "lookback_max",
                "validation_window_min", "validation_window_max", "htf_length_min",
                "htf_length_max", "stop_loss_min", "stop_loss_max", "velas_para_be_min",
                "velas_para_be_max", "tp_long_min", "tp_long_max", "tp_short_min",
                "tp_short_max", "max_losing_streak_min", "max_losing_streak_max",
                "cooldown_bars_min", "cooldown_bars_max", "max_reentries_min",
                "max_reentries_max", "max_post_reentries_min", "max_post_reentries_max"
            ]
            for key in range_keys:
                if key in window.key_dict:
                    window[key].update("")

            # Rellenar defaults de features activos
            fake_vals = {k: "" for k in FIELD_DEFAULTS}
            fill_defaults(window, fake_vals, "enable_high_condition", force=True)
            fill_defaults(window, fake_vals, "enable_low_condition", force=True)
            fill_defaults(window, fake_vals, "use_validation_window", force=True)

            # Resetear visibilidad de bloques
            window["col_rsi"].update(visible=False)
            window["col_adx"].update(visible=False)
            window["col_hl"].update(visible=True)
            window["col_valwin"].update(visible=True)
            window["col_htf"].update(visible=False)
            window["col_sl"].update(visible=False)
            window["col_be"].update(visible=False)
            window["col_tp"].update(visible=False)
            window["col_cooldown"].update(visible=False)
            window["col_reentry"].update(visible=False)
            window["col_post_reentry"].update(visible=False)

            # Habilitar checkboxes ON/OFF
            for key_on in ["use_rsi_long", "use_rsi_short", "use_adx_filter",
                           "enable_high_condition", "enable_low_condition",
                           "use_htf_filter", "use_stop_loss", "activar_stop_be",
                           "use_take_profit_long", "use_take_profit_short",
                           "enable_cooldown", "enable_reentry", "enable_post_crossover_entry"]:
                if key_on in window.key_dict:
                    window[key_on].update(disabled=False)

            # Resetear botones
            window["run"].update(disabled=False, button_color=(COLORS["text_primary"], COLORS["success"]))
            window["run_auto"].update(disabled=False, button_color=(COLORS["text_primary"], "#9b59b6"))
            window["stop"].update(disabled=True, button_color=(COLORS["text_primary"], COLORS["danger"]))
            window["ver_grafico"].update(disabled=True, button_color=(COLORS["text_primary"], COLORS["info"]))

            # <--- NUEVO: Resetear indicador de estado ---
            window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])

            # Limpiar output
            window["output"].update("")

            continue

        # --- Evento: Abrir carpeta de reportes ---
        if event == "open_reports":
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reportes")
            os.makedirs(reports_dir, exist_ok=True)
            subprocess.Popen(f'explorer "{os.path.abspath(reports_dir)}"')

        # --- Evento: Guardar preset ---
        if event == "save_preset":
            save_preset(values)

        # --- Evento: Cargar preset ---
        if event == "load_preset":
            load_preset(window)

        # --- Evento: Ejecutar Optimización ---
        if event == "run":
            source = values.get("data_source", "")
            timeframe_run = values.get("timeframe", "")
            symbol_run = values.get("symbol", "").strip()

            # Validación de acciones con timeframe no diario
            if source.startswith("Acciones") and timeframe_run != "1d":
                respuesta = sg.popup_yes_no(
                    f"⚠️ ADVERTENCIA CRÍTICA\n\n"
                    f"Estás intentando optimizar {symbol_run} (Acciones) con timeframe {timeframe_run}.\n\n"
                    f"Yahoo Finance tiene datos CORRUPTOS para timeframes intradiarios.\n"
                    f"Los resultados de la optimización NO serán confiables.\n\n"
                    f"¿Deseas continuar de todas formas?",
                    title="⚠️ Datos corruptos - Riesgo de resultados inválidos",
                    keep_on_top=True
                )
                if respuesta == "No":
                    print("\n[INFO] Optimización cancelada por timeframe no soportado para acciones.\n")
                    continue
                else:
                    print(f"\n[ADVERTENCIA] Usuario optó por continuar con {symbol_run} {timeframe_run} "
                          f"a pesar de la advertencia.\n")

            errores = []

            # Validaciones de formato según fuente de datos
            source = values["data_source"]
            symbol = values["symbol"].strip()

            if source.startswith("Cripto") and "/" not in symbol:
                errores.append("Para cripto usá formato XXX/USDT (ej: SOL/USDT).")
            if source.startswith("Acciones") and "/" in symbol:
                errores.append("Para acciones usá tickers sin barra (ej: AAPL, MSFT, SPY).")

            # Validaciones de campos obligatorios
            symbol_val = values.get("symbol", "").strip()
            if not symbol_val:
                errores.append("• Símbolo vacío (ej: SOL/USDT)")

            try:
                candles_val = int(values.get("candles", ""))
                if candles_val < 100:
                    errores.append("• Cantidad de velas debe ser al menos 100")
            except:
                errores.append("• Cantidad de velas inválida (debe ser un número entero)")

            try:
                trials_val = int(values.get("trials", ""))
                if trials_val < 1:
                    errores.append("• Cantidad de trials debe ser al menos 1")
            except:
                errores.append("• Cantidad de trials inválida (debe ser un número entero)")

            try:
                min_trades_val = int(values.get("min_trades", "30"))
                if min_trades_val < 1:
                    errores.append("• Mínimo de trades debe ser al menos 1")
            except:
                errores.append("• Mínimo de trades inválido (debe ser un número entero)")

            tiene_ma = any(values.get(k) for k in ["ma_ema", "ma_sma", "ma_wma", "ma_hma", "ma_dema"])
            if not tiene_ma:
                errores.append("• Seleccioná al menos un tipo de media móvil (EMA, SMA, etc.)")

            try:
                ma1_min_v = int(values.get("ma1_min", ""))
                ma1_max_v = int(values.get("ma1_max", ""))
                if ma1_min_v >= ma1_max_v:
                    errores.append("• Rango MA1: el mínimo debe ser menor que el máximo")
            except:
                errores.append("• Rango MA1 inválido (completá ambos campos con números enteros)")

            try:
                ma2_min_v = int(values.get("ma2_min", ""))
                ma2_max_v = int(values.get("ma2_max", ""))
                if ma2_min_v >= ma2_max_v:
                    errores.append("• Rango MA2: el mínimo debe ser menor que el máximo")
            except:
                errores.append("• Rango MA2 inválido (completá ambos campos con números enteros)")

            if errores:
                sg.popup_error("No se puede iniciar la optimización:\n\n" + "\n".join(errores),
                               title="Campos incompletos")
                continue

            # <--- NUEVO: Actualizar indicador de estado al iniciar optimización ---
            window["status_indicator"].update("⚙️ Optimizando...", text_color=COLORS["warning"])

            # Preparar para la optimización
            _stop_event.clear()
            window["run"].update(disabled=True, button_color=(COLORS["text_primary"], "#888888"))
            window["run_auto"].update(disabled=True, button_color=(COLORS["text_primary"], "#888888"))
            window["stop"].update(disabled=False, button_color=(COLORS["text_primary"], COLORS["danger"]))
            window['progress'].update(current_count=0.0)

            print("\n" + "━" * 55)
            print(f"  ▶  {symbol_val.upper()}  |  {values['timeframe']}  |  {trials_val} trials")
            print("━" * 55 + "\n")

            values["modo_optimizacion"] = values.get("modo_opt", "Personalizado")

            threading.Thread(
                target=optimization_thread_wrapper,
                args=(window, values, _stop_event),
                daemon=True
            ).start()


        # --- Evento: Ejecutar Optimización Automática ---------------------------------------------------------------
        if event == "run_auto":
            source = values.get("data_source", "")
            timeframe_run = values.get("timeframe", "")
            symbol_run = values.get("symbol", "").strip()
            
            # Advertencia para acciones con timeframe no diario
            if source.startswith("Acciones") and timeframe_run != "1d":
                respuesta = sg.popup_yes_no(
                    f"⚠️ ADVERTENCIA CRÍTICA\n\n"
                    f"Optimización automática con {symbol_run} (Acciones) y timeframe {timeframe_run}.\n\n"
                    f"Yahoo Finance tiene datos CORRUPTOS para timeframes intradiarios.\n"
                    f"¿Deseas continuar de todas formas?",
                    title="⚠️ Datos potencialmente corruptos",
                    keep_on_top=True
                )
                if respuesta == "No":
                    continue
            
            # Validar campos obligatorios
            errores = []
            symbol_val = values.get("symbol", "").strip()
            if not symbol_val:
                errores.append("• Símbolo vacío (ej: SOL/USDT)")
            
            try:
                candles_val = int(values.get("candles", ""))
                if candles_val < 100:
                    errores.append("• Cantidad de velas debe ser al menos 100")
            except:
                errores.append("• Cantidad de velas inválida (debe ser un número entero)")
            
            # Validar tipos de MA
            tiene_ma = any(values.get(k) for k in ["ma_ema", "ma_sma", "ma_wma", "ma_hma", "ma_dema"])
            if not tiene_ma:
                errores.append("• Seleccioná al menos un tipo de media móvil")
            
            if errores:
                sg.popup_error("No se puede iniciar la optimización automática:\n\n" + "\n".join(errores),
                               title="Campos incompletos")
                continue
            
            # Actualizar indicador de estado
            window["status_indicator"].update("🚀 Optimización Auto multi-fase...", text_color=COLORS["warning"])
            
            # Preparar para la optimización
            _stop_event.clear()
            window["run"].update(disabled=True, button_color=(COLORS["text_primary"], "#888888"))
            window["run_auto"].update(disabled=True, button_color=(COLORS["text_primary"], "#888888"))
            window["stop"].update(disabled=False, button_color=(COLORS["text_primary"], COLORS["danger"]))
            window['progress'].update(current_count=0.0)
            
            print("\n" + "━" * 55)
            print(f"  🚀 OPTIMIZACIÓN AUTOMÁTICA | {symbol_val.upper()} | {values['timeframe']}")
            print("━" * 55 + "\n")
            
            values["modo_optimizacion"] = "Automatico"
            
            # Lanzar hilo con la optimización automática
            threading.Thread(
                target=optimization_thread_wrapper_auto,
                args=(window, values, _stop_event),
                daemon=True
            ).start()

        #-------------------------------------------------------------------------------------------------------------------------------------------

        # --- Evento: Verificar velas ---
        if event == "check_candles":
            # <--- NUEVO: Actualizar indicador de estado ---
            window["status_indicator"].update("🔍 Verificando velas...", text_color=COLORS["info"])
            
            try:
                symbol = values["symbol"].strip()
                timeframe = values["timeframe"]
                requested = values["candles"].strip()
                source = values["data_source"]

                # Validación con advertencia para acciones intradiarias
                if source.startswith("Acciones") and timeframe != "1d":
                    respuesta = sg.popup_yes_no(
                        f"⚠️ ADVERTENCIA: Datos potencialmente corruptos\n\n"
                        f"Estás verificando {symbol} (Acciones) con timeframe {timeframe}.\n\n"
                        f"Yahoo Finance tiene datos CORRUPTOS para timeframes intradiarios.\n"
                        f"Los resultados de la verificación podrían ser INCORRECTOS.\n\n"
                        f"¿Deseas continuar de todas formas?",
                        title="⚠️ Datos potencialmente corruptos",
                        keep_on_top=True
                    )
                    if respuesta == "No":
                        print("\n[INFO] Verificación cancelada por timeframe no recomendado para acciones.\n")
                        # <--- NUEVO: Restaurar estado si cancela ---
                        window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])
                        continue
                    else:
                        print(f"\n[ADVERTENCIA] Usuario optó por verificar {symbol} {timeframe} "
                              f"a pesar de la advertencia.\n")

                if not symbol or not requested:
                    sg.popup_error("Completá Símbolo y Cantidad de velas antes de verificar.",
                                   title="Campos incompletos")
                    # <--- NUEVO: Restaurar estado ---
                    window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])
                    continue

                requested = int(requested)
                print(f"\nVerificando disponibilidad de velas en {source}...\n")

                res = check_available_candles(symbol=symbol, timeframe=timeframe,
                                              requested=requested, data_source=source)

                if res["available"] == 0:
                    print(f"⚠️ El proveedor de datos ({source}) no devolvió velas para el símbolo {symbol}.")
                    sg.popup_error(f"No se recibieron datos para {symbol}. Revisa el ticker o el timeframe.",
                                   title="Sin datos")
                else:
                    reporte_velas = (
                        f"Exchange: {res['exchange']}\n"
                        f"Activo: {res['symbol']}\n"
                        f"Nombre: {res['name']}\n"
                        f"Ultimo precio: $ {res['last_price']:.2f}\n"
                        f"Cantidad de velas que se devolverían: {res['available']}\n"
                        f"Rango a obtener: {res['start']} → {res['end']}\n"
                    )
                    print(reporte_velas)

            except ValueError:
                sg.popup_error("Cantidad de velas debe ser un número entero válido.", title="Error de formato")
            except Exception as e:
                print("Error al verificar disponibilidad de velas:", e)
            
            # <--- NUEVO: Restaurar estado después de verificar ---
            window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])

        # --- Evento: Preview de velas ---
        if event == "preview_velas":
            # <--- NUEVO: Actualizar indicador de estado ---
            window["status_indicator"].update("📊 Generando preview...", text_color=COLORS["info"])
            
            symbol_prev = values.get("symbol", "").strip().upper()
            tf_prev = values.get("timeframe", "5m")
            candles_prev = values.get("candles", "").strip()
            source_prev = values["data_source"]

            if not symbol_prev or not candles_prev:
                sg.popup_error("Completá Símbolo y Cantidad de velas antes de hacer el preview.",
                               title="Campos incompletos")
                # <--- NUEVO: Restaurar estado ---
                window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])
            else:
                try:
                    candles_prev = int(candles_prev)
                except:
                    sg.popup_error("Cantidad de velas inválida.", title="Error")
                    candles_prev = None
                    # <--- NUEVO: Restaurar estado ---
                    window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])

                if candles_prev:
                    window["preview_velas"].update(disabled=True, button_color=("white", "gray"))
                    print(f"\n[PREVIEW] Generando gráfico para {symbol_prev} {tf_prev} ({candles_prev:,} velas)...\n")

                    def _preview_thread(s=symbol_prev, t=tf_prev, c=candles_prev, src=source_prev):
                        html = generar_preview_velas(symbol=s, timeframe=t, total_candles=c, data_source=src)
                        window.write_event_value("-PREVIEW_LISTO-", html)

                    threading.Thread(target=_preview_thread, daemon=True).start()

        # --- Evento: Preview listo ---
        if event == "-PREVIEW_LISTO-":
            # <--- NUEVO: Restaurar estado cuando el preview termina ---
            window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])
            window["preview_velas"].update(disabled=False, button_color=("white", "#2d6a8a"))
            html_prev = values["-PREVIEW_LISTO-"]
            if html_prev:
                webbrowser.open(f"file:///{html_prev}")
            else:
                sg.popup_error("No se pudo generar el preview. Revisá el output para más detalles.",
                               title="Error")

        # --- Evento: Calcular trials recomendados ---
        if event == "calcular_trials":
            # <--- NUEVO: Actualizar indicador de estado ---
            window["status_indicator"].update("📐 Calculando espacio de búsqueda...", text_color=COLORS["info"])
            try:
                config = build_config(values)
                reporte = calcular_trials_recomendados(values, config)
                print(reporte)
            except Exception as e:
                print(f"\n[ERROR] No se pudo calcular los trials recomendados: {e}")
            # <--- NUEVO: Restaurar estado ---
            window["status_indicator"].update("● Sistema listo", text_color=COLORS["success"])

        # --- Evento: Cargar Semilla ---
        if event == "load_semilla":
            ruta = sg.popup_get_file(
                "Seleccioná un archivo Semilla JSON",
                file_types=(("JSON Files", "*.json"),),
                no_window=True
            )
            if ruta:
                try:
                    with open(ruta, "r", encoding="utf-8") as f:
                        semilla = json.load(f)

                    gui_vals = semilla.get("gui_values", {})
                    for k, v in gui_vals.items():
                        if k in window.key_dict:
                            window[k].update(v)

                    window["symbol"].update(semilla.get("symbol", ""))
                    window["timeframe"].update(semilla.get("timeframe", ""))

                    opt = semilla.get("optuna_settings", {})
                    if "trials" in opt:
                        window["trials"].update(opt["trials"])
                    if "multi_run" in opt:
                        window["multi_run"].update(opt["multi_run"])
                    if "runs" in opt:
                        window["multi_runs_count"].update(opt["runs"])
                    if "mode" in opt:
                        window["modo_opt"].update(opt["mode"])

                    sg.popup("Semilla cargada correctamente.\nEl GUI fue actualizado.")
                    # <--- NUEVO: Indicar éxito ---
                    window["status_indicator"].update("✅ Semilla cargada", text_color=COLORS["success"])

                except Exception as e:
                    sg.popup_error(f"Error al cargar la Semilla:\n{e}")
                    # <--- NUEVO: Indicar error ---
                    window["status_indicator"].update("❌ Error al cargar semilla", text_color=COLORS["danger"])

    window.close()


# ============================================================================
# SECCIÓN 7: PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    gui_main()