"""
Gráficos Interactivos - Generación de gráficos con Plotly
==========================================================
Funciones para generar gráficos de velas, trades y curvas de equity.
"""

import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importaciones del proyecto
from config import CONSTANTS

# Directorio de reportes (definido aquí para evitar import circular)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "reportes")
os.makedirs(output_dir, exist_ok=True)


def generar_grafico(df, trades, params, symbol, timeframe):
    """
    Genera un gráfico interactivo Plotly con velas OHLC, MA1, MA2,
    markers de entrada/salida y equity curve.
    Guarda como HTML y retorna el path.
    """
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
        )

        # --- Velas ---
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'],   close=df['close'],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # --- MAs ---
        if 'ma1' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ma1'],
                name=f"MA1 ({params.get('ma1_type','')}-{params.get('ma1_length','')})",
                line=dict(color="#FFA500", width=1.5),
            ), row=1, col=1)

        if 'ma2' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ma2'],
                name=f"MA2 ({params.get('ma2_type','')}-{params.get('ma2_length','')})",
                line=dict(color="#1E90FF", width=1.5),
            ), row=1, col=1)

        # --- Trades ---
        for t in trades:
            es_long = t['dir'] == "LONG"
            es_ganador = t['net_pnl'] > 0
            ret_pct = (t['net_pnl'] / (t['entry_price'] * t['size'])) * 100 if t['entry_price'] and t['size'] else 0

            color_entrada = "#00C853" if es_long else "#D50000"
            color_salida = "#00C853" if es_ganador else "#FF6D00"
            simbolo_entrada = "triangle-up" if es_long else "triangle-down"

            label_entrada = f"{'LONG' if es_long else 'SHORT'}<br>Entrada: {t['entry_price']:.4f}"
            label_salida = (
                f"{'✅ WIN' if es_ganador else '❌ LOSS'}<br>"
                f"Salida: {t['exit_price']:.4f}<br>"
                f"PnL: {t['net_pnl']:.2f} ({ret_pct:+.2f}%)<br>"
                f"Razón: {t.get('reason','')}"
            )

            fig.add_trace(go.Scatter(
                x=[t['entry_time']], y=[t['entry_price']],
                mode='markers',
                marker=dict(symbol=simbolo_entrada, color=color_entrada, size=12,
                            line=dict(width=1, color='white')),
                name="", showlegend=False,
                hovertext=label_entrada, hoverinfo="text",
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[t['exit_time']], y=[t['exit_price']],
                mode='markers',
                marker=dict(symbol="circle", color=color_salida, size=10,
                            line=dict(width=1, color='white')),
                name="", showlegend=False,
                hovertext=label_salida, hoverinfo="text",
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[t['entry_time'], t['exit_time']],
                y=[t['entry_price'], t['exit_price']],
                mode='lines',
                line=dict(color=color_salida, width=1, dash='dot'),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

        # --- Equity curve ---
        equity_vals = [CONSTANTS["initial_capital"]]
        for t in trades:
            equity_vals.append(equity_vals[-1] + t['net_pnl'])

        fig.add_trace(go.Scatter(
            x=[t['exit_time'] for t in trades],
            y=equity_vals[1:],
            fill='tozeroy',
            name="Equity",
            line=dict(color="#7B1FA2", width=1.5),
            fillcolor="rgba(123,31,162,0.15)",
        ), row=2, col=1)

        fig.update_layout(
            title=f"{symbol} {timeframe} — Backtest estrategia optimizada",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=800,
        )
        fig.update_yaxes(title_text="Precio", row=1, col=1)
        fig.update_yaxes(title_text="Equity $", row=2, col=1)

        clean_symbol = symbol.replace("/", "").replace(":", "")
        timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
        html_path = os.path.join(output_dir, f"{timestamp} Grafico {clean_symbol}_{timeframe}.html")

        fig.write_html(html_path)
        print(f"\n[ÉXITO] Gráfico generado:\n  {html_path}")
        return html_path

    except Exception as e:
        print(f"\n[ERROR] No se pudo generar el gráfico: {e}")
        return None


def generar_preview_velas(symbol, timeframe, total_candles, data_source):
    """
    Descarga (o usa caché) el OHLC y genera un gráfico de velas
    limpio en el browser — sin trades ni MAs, solo para evaluación visual.
    Soporta tanto Cripto (Binance) como Acciones (Yahoo Finance).
    Retorna el path del HTML generado o None si falla.
    """
    try:
        # Mostrar advertencia para acciones con timeframe intradiario
        if data_source.startswith("Acciones") and timeframe != "1d":
            print(f"\n[ADVERTENCIA] Preview para {symbol} {timeframe} (Acciones)")
            print(f"            Yahoo Finance puede tener datos corruptos para timeframes intradiarios.")
            print(f"            El gráfico puede mostrar velas incorrectas.\n")

        print(f"\n[PREVIEW] Descargando velas para {symbol} {timeframe} ({total_candles} velas) desde {data_source}...")

        # Importación local para evitar circular import
        from optimizador_main import get_data_efficiently
        df = get_data_efficiently(symbol, timeframe, total_candles, data_source)

        if df is None or df.empty:
            print(f"[ERROR PREVIEW] No se pudieron obtener datos para el gráfico de preview.")
            return None

        df = df.dropna()

        fecha_inicio = df.index[0].strftime("%Y-%m-%d %H:%M")
        fecha_fin = df.index[-1].strftime("%Y-%m-%d %H:%M")

        # Agregar nota de advertencia en el título si es acciones intradiario
        titulo = f"{symbol} {timeframe} — Preview de velas ({data_source})"
        if data_source.startswith("Acciones") and timeframe != "1d":
            titulo += " ⚠️ ADVERTENCIA: Datos pueden ser corruptos"

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'],   close=df['close'],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ))

        fig.update_layout(
            title=(
                f"{titulo}<br>"
                f"<sup>{len(df):,} velas reales  |  {fecha_inicio} → {fecha_fin}</sup>"
            ),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode="x unified",
            height=700,
            xaxis_title="Fecha",
            yaxis_title="Precio",
        )

        clean_symbol = symbol.replace("/", "").replace(":", "")
        timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
        html_path = os.path.join(output_dir, f"{timestamp} Preview {clean_symbol}_{timeframe}.html")

        fig.write_html(html_path)
        print(f"[PREVIEW] Gráfico guardado en:\n  {html_path}")
        return html_path

    except Exception as e:
        print(f"\n[ERROR] No se pudo generar el preview: {e}")
        return None