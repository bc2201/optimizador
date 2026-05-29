"""
OPTIMIZADOR AUTOMÁTICO MULTI-FASE
=================================
Sistema que converge automáticamente hacia la mejor configuración
mediante 3 fases: Exploración → Refinamiento → Validación.

Uso:
    from optimization import ejecutar_optimizacion_automatica
    
    resultados = ejecutar_optimizacion_automatica(
        df=df,
        config_base=config,
        features=features,
        symbol="BTC/USDT",
        timeframe="1h"
    )
"""

import numpy as np
import threading
from optimizador_main import run_single_optuna, run_backtest
from backtest import calcular_drawdown_maximo 


class OptimizadorAutomatico:
    """
    Optimizador que ejecuta automáticamente 3 fases de optimización.
    
    Fase 1: Exploración Rápida (paralelo, pocos trials)
    Fase 2: Refinamiento (serie, rangos acotados, más trials)
    Fase 3: Validación (multi-run, confirma robustez)
    """
    
    def __init__(self, df, config_base, features, symbol, timeframe, 
            config_fases=None, config_convergencia=None, 
            config_metricas=None, gui_values=None, 
            df_train=None, df_test=None, constants=None, verbose=True):
        """
        Args:
            df: DataFrame con datos históricos completos
            df_train: DataFrame para entrenamiento (IS) - opcional
            df_test: DataFrame para validación (OOS) - opcional
            constants: Constantes de configuración
            ... otros args ...
        """

        self.df_train = df_train if df_train is not None else df
        self.df_test = df_test if df_test is not None else df
        self.constants = constants if constants else {}
        self.gui_values = gui_values if gui_values else {}
        self.df = df
        self.config_base = config_base.copy()
        self.features = features
        self.symbol = symbol
        self.timeframe = timeframe
        self.verbose = verbose
        
        # Cargar configuraciones (usar defaults si no se proveen)
        try:
            from config_auto_optimizer import (CONFIG_FASES, CONFIG_CONVERGENCIA, CONFIG_METRICAS)
            
            self.config_fases = config_fases if config_fases else CONFIG_FASES
            self.config_convergencia = config_convergencia if config_convergencia else CONFIG_CONVERGENCIA
            self.config_metricas = config_metricas if config_metricas else CONFIG_METRICAS
        except ImportError:
            # Si no existe config_auto_optimizer.py, usar valores por defecto
            self.config_fases = config_fases if config_fases else self._default_fases()
            self.config_convergencia = config_convergencia if config_convergencia else self._default_convergencia()
            self.config_metricas = config_metricas if config_metricas else self._default_metricas()
        
        # Historial de resultados
        self.historial = []
    
    def _default_fases(self):
        """Valores por defecto si no existe config_auto_optimizer.py"""
        return {
            "fase_1": {"trials": 2000, "modo": "paralelo"},
            "fase_2": {"trials": 1500, "modo": "serie"},
            "fase_3": {"trials_por_corrida": 800, "corridas": 5, "modo": "serie"}
        }
    
    def _default_convergencia(self):
        """Valores por defecto para convergencia"""
        return {
            "activar": True,
            "ventana": 75,
            "tolerancia": 0.002,
            "trials_minimos": 400,
            "mejor_score_minimo": 0.85
        }
    
    def _default_metricas(self):
        """Valores por defecto para métricas"""
        return {
            "fase_1": {"use_pf": True, "peso_pf": 60.0, "use_winrate": True, "peso_winrate": 40.0,
                      "use_drawdown": False, "peso_drawdown": 0.0, "use_n_trades": False, 
                      "peso_n_trades": 0.0, "min_trades": 15},
            "fase_2": {"use_pf": True, "peso_pf": 40.0, "use_winrate": True, "peso_winrate": 30.0,
                      "use_drawdown": True, "peso_drawdown": 20.0, "use_n_trades": True, 
                      "peso_n_trades": 10.0, "min_trades": 20},
            "fase_3": {"use_pf": True, "peso_pf": 35.0, "use_winrate": True, "peso_winrate": 30.0,
                      "use_drawdown": True, "peso_drawdown": 35.0, "use_n_trades": False, 
                      "peso_n_trades": 0.0, "min_trades": 30}
        }
        
    def _log(self, mensaje):
        """Muestra mensaje si verbose está activado"""
        if self.verbose:
            print(mensaje)
    
    def _acotar_rangos(self, params_referencia):
        """
        Reduce los rangos de búsqueda alrededor de los parámetros encontrados.
        Redondea los valores a 1 decimal para evitar warnings de Optuna.
        """
        config_acotada = self.config_base.copy()
        
        # Acotar MA lengths (±30% alrededor del valor encontrado)
        if "ma1_length" in params_referencia:
            ma1 = params_referencia["ma1_length"]
            config_acotada["ma1_min"] = max(2, int(ma1 * 0.7))
            config_acotada["ma1_max"] = int(ma1 * 1.3)
        
        if "ma2_length" in params_referencia:
            ma2 = params_referencia["ma2_length"]
            config_acotada["ma2_min"] = max(5, int(ma2 * 0.7))
            config_acotada["ma2_max"] = int(ma2 * 1.3)
        
        # Acotar RSI si existe
        if "rsi_length" in params_referencia:
            rsi = params_referencia["rsi_length"]
            rango_actual = self.config_base.get("rsi_length_range", (2, 50))
            nuevo_min = max(2, int(rsi * 0.7))
            nuevo_max = min(rango_actual[1], int(rsi * 1.3))
            if nuevo_min < nuevo_max:
                config_acotada["rsi_length_range"] = (nuevo_min, nuevo_max)
        
        # ============================================================
        # REDONDEAR RANGOS DECIMALES A MÚLTIPLOS DE 0.1
        # ============================================================
        
        # Acotar stop loss (redondear a 1 decimal)
        if "stop_loss_pct" in params_referencia:
            sl = params_referencia["stop_loss_pct"]
            rango_actual = self.config_base.get("sl_range", (0.3, 10.0))
            nuevo_min = round(max(0.3, sl * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], sl * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["sl_range"] = (nuevo_min, nuevo_max)
        
        # Acotar take profit long (redondear a 1 decimal)
        if "tp_long_pct" in params_referencia:
            tp = params_referencia["tp_long_pct"]
            rango_actual = self.config_base.get("tp_long_range", (0.3, 99.0))
            nuevo_min = round(max(0.3, tp * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], tp * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["tp_long_range"] = (nuevo_min, nuevo_max)
        
        # Acotar take profit short (redondear a 1 decimal)
        if "tp_short_pct" in params_referencia:
            tp = params_referencia["tp_short_pct"]
            rango_actual = self.config_base.get("tp_short_range", (0.3, 99.0))
            nuevo_min = round(max(0.3, tp * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], tp * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["tp_short_range"] = (nuevo_min, nuevo_max)
        
        # Acotar ADX threshold (redondear a 1 decimal)
        if "adx_threshold" in params_referencia:
            adx = params_referencia["adx_threshold"]
            rango_actual = self.config_base.get("adx_thr_range", (5.0, 70.0))
            nuevo_min = round(max(5.0, adx * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], adx * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["adx_thr_range"] = (nuevo_min, nuevo_max)
        
        # Acotar RSI min (redondear a 1 decimal)
        if "rsi_min" in params_referencia:
            rsi_min = params_referencia["rsi_min"]
            rango_actual = self.config_base.get("rsi_min_range", (50.0, 80.0))
            nuevo_min = round(max(50.0, rsi_min * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], rsi_min * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["rsi_min_range"] = (nuevo_min, nuevo_max)
        
        # Acotar RSI max (redondear a 1 decimal)
        if "rsi_max" in params_referencia:
            rsi_max = params_referencia["rsi_max"]
            rango_actual = self.config_base.get("rsi_max_range", (5.0, 50.0))
            nuevo_min = round(max(5.0, rsi_max * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], rsi_max * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["rsi_max_range"] = (nuevo_min, nuevo_max)
    
        return config_acotada
    
    def _metrics_config_para_fase(self, fase):
        """Usa la configuración de métricas desde self.config_metricas"""
        return self.config_metricas.get(f"fase_{fase}", {})
    
    def _fase_exploracion_rapida(self):
        """
        FASE 1: Exploración rápida en paralelo con detección de convergencia.
        Objetivo: Descartar malas configuraciones, identificar regiones prometedoras.
        """
        cfg_fase = self.config_fases.get("fase_1", {})
        n_trials = cfg_fase.get("trials", 2000)
        modo_paralelo = cfg_fase.get("modo", "paralelo") == "paralelo"
        
        self._log("\n" + "="*60)
        self._log("🔎 FASE 1: Exploración Rápida")
        
        # Mostrar configuración de convergencia si está activada
        if self.config_convergencia.get("activar", True):
            self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | Trials Max: {n_trials} | Convergencia: Sí")
            self._log(f"   Convergencia: ventana={self.config_convergencia['ventana']}, "
                     f"tolerancia={self.config_convergencia['tolerancia']*100:.1f}%, "
                     f"trials_min={self.config_convergencia['trials_minimos']}")
        else:
            self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | Trials: {n_trials} | Convergencia: No")
        self._log("="*60)
        
        metrics_config = self._metrics_config_para_fase(1)
        
        # ============================================================
        # OPTIMIZACIÓN CON O SIN CONVERGENCIA
        # ============================================================
        
        if self.config_convergencia.get("activar", True):
            # Con detección de convergencia
            import optuna
            
            convergencia_detectada = False
            trial_convergencia = 0
            
            def callback_convergencia(study, trial):
                nonlocal convergencia_detectada, trial_convergencia

                # No evaluar antes de trials_minimos
                if trial.number < self.config_convergencia["trials_minimos"]:
                    return

                # Evaluar cada 10 trials
                if trial.number % 10 != 0:
                    return

                ventana = self.config_convergencia["ventana"]
                if len(study.trials) < ventana:
                    return

                # Calcular mejora en la ventana - FILTRAR valores None
                ultimos_scores = []
                for t in study.trials[-ventana:]:
                    if t.value is not None:
                        ultimos_scores.append(t.value)
                
                # Si no hay suficientes valores válidos, salir
                if len(ultimos_scores) < ventana // 2:
                    return
                    
                mejor_en_ventana = max(ultimos_scores)
                peor_en_ventana = min(ultimos_scores)
                
                # Evitar división por cero
                if peor_en_ventana == 0:
                    mejora = 0
                else:
                    mejora = (mejor_en_ventana - peor_en_ventana) / abs(peor_en_ventana)
                
                tolerancia = self.config_convergencia["tolerancia"]
                
                if mejora < tolerancia:
                    convergencia_detectada = True
                    trial_convergencia = trial.number
                    study.stop()
            
            # Crear estudio con callback
            sampler = optuna.samplers.TPESampler(seed=None)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            
            def objective_wrapper(trial):
                from optimizador_main import objective
                return objective(trial, self.df, self.config_base, self.features, 
                                metrics_config, {}, threading.Lock())
            
            study.optimize(
                objective_wrapper,
                n_trials=n_trials,
                n_jobs=-1 if modo_paralelo else 1,
                callbacks=[callback_convergencia]
            )
            
            best_score = study.best_value if study.best_value else 0
            best_params = study.best_params if study.best_params else {}
            
            trials_usados = trial_convergencia if convergencia_detectada else n_trials
            
            if convergencia_detectada:
                self._log(f"\n   ⏹️  Detenido en trial {trials_usados} de {n_trials} (convergencia)")
            else:
                self._log(f"\n   ⏹️  Completados {n_trials} trials (límite alcanzado)")
            
        else:
            # Sin convergencia (método original)
            pf, best_params = run_single_optuna(
                df=self.df,
                config=self.config_base,
                n_trials=n_trials,
                modo_paralelo=modo_paralelo,
                features=self.features,
                metrics_config=metrics_config
            )
            best_score = pf
            trials_usados = n_trials
        
        self.historial.append({
            "fase": 1,
            "nombre": "Exploración Rápida",
            "best_score": best_score,
            "best_params": best_params,
            "trials": trials_usados,
            "trials_max": n_trials,
            "modo": "paralelo" if modo_paralelo else "serie",
            "convergencia_temprana": self.config_convergencia.get("activar", False)
        })
        
        self._log(f"\n   ✅ Mejor score: {best_score:.4f}")
        self._log(f"   📊 Parámetros encontrados: {len(best_params)}")
        
        return best_params
    
    def _fase_refinamiento(self, params_iniciales):
        """
        FASE 2: Refinamiento en serie con rangos acotados.
        Objetivo: Afinar los mejores parámetros.
        """
        cfg_fase = self.config_fases.get("fase_2", {})
        n_trials = cfg_fase.get("trials", 1500)
        modo_paralelo = cfg_fase.get("modo", "serie") == "paralelo"
        
        self._log("\n" + "="*60)
        self._log("🎯 FASE 2: Refinamiento")
        self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | Trials: {n_trials} | Rangos: Acotados")
        self._log("="*60)
        
        # Acotar rangos alrededor de los parámetros encontrados
        config_refinada = self._acotar_rangos(params_iniciales)
        metrics_config = self._metrics_config_para_fase(2)
        
        pf, best_params = run_single_optuna(
            df=self.df,
            config=config_refinada,
            n_trials=n_trials,
            modo_paralelo=modo_paralelo,
            features=self.features,
            metrics_config=metrics_config
        )
        
        self.historial.append({
            "fase": 2,
            "nombre": "Refinamiento",
            "best_score": pf,
            "best_params": best_params,
            "trials": n_trials,
            "modo": "paralelo" if modo_paralelo else "serie",
            "rangos_acotados": True
        })
        
        mejora = (pf - self.historial[0]['best_score']) * 100 if self.historial[0]['best_score'] > 0 else 0
        self._log(f"\n   ✅ Mejor score: {pf:.4f}")
        self._log(f"   📈 Mejora desde fase 1: {mejora:.1f}%")
        
        return best_params
    
    def _fase_validacion(self, params_optimos):
        """
        FASE 3: Validación con multi-run.
        Objetivo: Confirmar robustez y evitar overfitting.
        """
        cfg_fase = self.config_fases.get("fase_3", {})
        trials_por_corrida = cfg_fase.get("trials_por_corrida", 800)
        corridas = cfg_fase.get("corridas", 5)
        modo_paralelo = cfg_fase.get("modo", "serie") == "paralelo"
        
        self._log("\n" + "="*60)
        self._log("🛡️ FASE 3: Validación")
        self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | {corridas} corridas de {trials_por_corrida} trials")
        self._log("="*60)
        
        metrics_config = self._metrics_config_para_fase(3)
        
        # Usar rangos acotados alrededor de params_optimos
        config_validacion = self._acotar_rangos(params_optimos)
        
        resultados_validacion = []
        mejores_params = params_optimos
        mejor_score = 0
        
        # Variables para guardar métricas de la mejor corrida
        mejor_pf_train = 0
        mejor_pf_test = 0
        mejor_winrate_train = 0
        mejor_winrate_test = 0
        mejor_drawdown_train = 0
        mejor_drawdown_test = 0
        mejor_trades_train = 0
        mejor_trades_test = 0
        
        for run in range(corridas):
            self._log(f"\n   🔄 Corrida {run+1}/{corridas}...")
            
            # Optimizar sobre datos de entrenamiento (IS)
            pf_train_score, params = run_single_optuna(
                df=self.df_train,  # Necesitas tener df_train y df_test disponibles
                config=config_validacion,
                n_trials=trials_por_corrida,
                modo_paralelo=modo_paralelo,
                features=self.features,
                metrics_config=metrics_config
            )
            
            # Traducir parámetros para backtest
            MAPEO_AUTO_BACKTEST = {
                "usar_rsi_long": "use_rsi_long",
                "usar_rsi_short": "use_rsi_short",
                "usar_adx": "use_adx_filter",
                "usar_high": "enable_high_condition",
                "usar_low": "enable_low_condition",
                "usar_htf": "use_htf_filter",
                "usar_sl": "use_stop_loss",
                "usar_be": "activar_stop_be",
                "usar_tp_long": "use_take_profit_long",
                "usar_tp_short": "use_take_profit_short",
                "usar_cooldown": "enable_cooldown",
                "usar_reentry": "enable_reentry",
                "usar_post_re": "enable_post_crossover_entry"
            }
            
            cleaned_params = {}
            for k, v in params.items():
                if k in MAPEO_AUTO_BACKTEST:
                    cleaned_params[MAPEO_AUTO_BACKTEST[k]] = v
                else:
                    cleaned_params[k] = v
            
            cleaned_features = {k: v for k, v in self.features.items() if v != "auto"}
            
            # Ejecutar backtest en TRAIN para obtener métricas reales
            pf_train_backtest, equity_train, trades_train = run_backtest(
                self.df_train,
                **cleaned_params,
                **cleaned_features,
                **self.constants
            )
            
            # Calcular métricas de TRAIN
            winrate_train = len([t for t in trades_train if t['net_pnl'] > 0]) / len(trades_train) * 100 if trades_train else 0
            drawdown_train = calcular_drawdown_maximo(list(equity_train)) if len(equity_train) > 0 else 0
            n_trades_train = len(trades_train)
            
            # Validación sobre TEST (OOS)
            pf_oos, equity_test, trades_test = run_backtest(
                self.df_test,
                **cleaned_params,
                **cleaned_features,
                **self.constants
            )
            
            # Calcular métricas de TEST
            winrate_test = len([t for t in trades_test if t['net_pnl'] > 0]) / len(trades_test) * 100 if trades_test else 0
            drawdown_test = calcular_drawdown_maximo(list(equity_test)) if len(equity_test) > 0 else 0
            n_trades_test = len(trades_test)
            
            resultados_validacion.append(pf_oos)
            
            # Guardar la mejor corrida por pf_oos
            if pf_oos > mejor_score:
                mejor_score = pf_oos
                mejores_params = params
                mejor_pf_train = pf_train_backtest
                mejor_pf_test = pf_oos
                mejor_winrate_train = winrate_train
                mejor_winrate_test = winrate_test
                mejor_drawdown_train = drawdown_train
                mejor_drawdown_test = drawdown_test
                mejor_trades_train = n_trades_train
                mejor_trades_test = n_trades_test
                
                self._log(f"      🆕 Nuevo mejor score: {pf_oos:.4f}")
        
        # Estadísticas de validación
        pf_mean = np.mean(resultados_validacion)
        pf_std = np.std(resultados_validacion)
        estabilidad = 1.0 - (pf_std / pf_mean) if pf_mean > 0 else 0
        
        self.historial.append({
            "fase": 3,
            "nombre": "Validación",
            "best_score": pf_mean,
            "best_params": mejores_params,
            "estabilidad": estabilidad,
            "resultados_individuales": resultados_validacion,
            "trials_por_corrida": trials_por_corrida,
            "corridas": corridas,
            "modo": "paralelo" if modo_paralelo else "serie",
            # ============================================================
            # NUEVO: Guardar métricas de Train y Test de la mejor corrida
            # ============================================================
            "pf_train": mejor_pf_train,
            "pf_test": mejor_pf_test,
            "winrate_train": mejor_winrate_train,
            "winrate_test": mejor_winrate_test,
            "drawdown_train": mejor_drawdown_train,
            "drawdown_test": mejor_drawdown_test,
            "trades_train": mejor_trades_train,
            "trades_test": mejor_trades_test
        })
        
        self._log(f"\n   ✅ Score promedio: {pf_mean:.4f} (±{pf_std:.4f})")
        self._log(f"   📊 Estabilidad: {estabilidad:.1%}")
        
        return mejores_params



    def _generar_reporte_txt(self, best_params, resultado_final, trades_final=None, equity_curve_final=None, pf_final=None):
        """
        Genera un reporte TXT con el formato unificado para optimización automática.
        Incluye resumen por fase, tabla de métricas y validación IS/OOS.
        
        Args:
            best_params: Mejores parámetros encontrados
            resultado_final: Diccionario con resultados de la fase 3
            trades_final: Lista de trades del backtest final (opcional)
            equity_curve_final: Serie de equity del backtest final (opcional)
            pf_final: Profit Factor final (opcional)
        """
        from datetime import datetime
        import os
        from backtest import calcular_drawdown_maximo
        
        # Crear carpeta de reportes si no existe
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reportes")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar timestamp y nombre de archivo
        timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        filename = f"{timestamp} AutoOptim_{safe_symbol}_{self.timeframe}.txt"
        filepath = os.path.join(output_dir, filename)
        
        now = datetime.now()
        fecha = now.strftime("%d/%m/%Y")
        hora = now.strftime("%H:%M")
        
        # Extraer datos básicos
        symbol = self.symbol
        timeframe = self.timeframe
        velas = len(self.df)
        
        # ============================================================
        # CALCULAR MÉTRICAS FINALES (si se pasaron los datos)
        # ============================================================
        if trades_final is not None and equity_curve_final is not None:
            n_trades = len(trades_final)
            if n_trades > 0:
                gross_profit = sum(t["net_pnl"] for t in trades_final if t["net_pnl"] > 0)
                gross_loss = -sum(t["net_pnl"] for t in trades_final if t["net_pnl"] < 0)
                if gross_loss == 0:
                    profit_factor = float("inf") if gross_profit > 0 else 0.0
                else:
                    profit_factor = gross_profit / gross_loss
                
                initial_capital = equity_curve_final.iloc[0] if len(equity_curve_final) > 0 else 0.0
                final_capital = equity_curve_final.iloc[-1] if len(equity_curve_final) > 0 else 0.0
                total_return = ((final_capital / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
                
                wins = [t for t in trades_final if t["net_pnl"] > 0]
                winrate = (len(wins) / n_trades) * 100.0
                
                max_dd = calcular_drawdown_maximo(list(equity_curve_final)) if len(equity_curve_final) > 0 else 0.0
                
                longs = [t for t in trades_final if t["dir"] == "LONG"]
                shorts = [t for t in trades_final if t["dir"] == "SHORT"]
                longs_total = len(longs)
                shorts_total = len(shorts)
                longs_win = len([t for t in longs if t["net_pnl"] > 0])
                shorts_win = len([t for t in shorts if t["net_pnl"] > 0])
                
                if longs_total > 0:
                    cap0 = initial_capital
                    cap_long = cap0 + sum(t["net_pnl"] for t in longs)
                    longs_ret = ((cap_long / cap0) - 1.0) * 100.0
                else:
                    longs_ret = 0.0
                
                if shorts_total > 0:
                    cap0 = initial_capital
                    cap_short = cap0 + sum(t["net_pnl"] for t in shorts)
                    shorts_ret = ((cap_short / cap0) - 1.0) * 100.0
                else:
                    shorts_ret = 0.0
            else:
                profit_factor = 0.0
                total_return = 0.0
                winrate = 0.0
                max_dd = 0.0
                longs_total = shorts_total = longs_win = shorts_win = 0
                longs_ret = shorts_ret = 0.0
                n_trades = 0
        else:
            # Valores por defecto si no se pasaron los datos
            profit_factor = pf_final if pf_final else resultado_final.get('best_score', 0)
            total_return = 0.0
            winrate = 0.0
            max_dd = 0.0
            longs_total = shorts_total = longs_win = shorts_win = 0
            longs_ret = shorts_ret = 0.0
            n_trades = 0
        
        best_score = resultado_final.get('best_score', 0)
        estabilidad = resultado_final.get('estabilidad', 0)
        
        # ============================================================
        # CONSTRUIR REPORTE
        # ============================================================
        lines = []
        
        # Encabezado principal
        lines.append("=" * 77)
        lines.append(f"🚀 OPTIMIZACIÓN AUTOMÁTICA\t|\tReporte de Optimización")
        lines.append("=" * 77)
        lines.append("")
        lines.append("-" * 77)
        lines.append(f"📅 Fecha: {fecha} {hora}\t-\t🔧 Versión optimizador: 19")
        lines.append("-" * 77)
        lines.append("")
        lines.append(f"📊 Activo: {symbol}\t⏱️ Timeframe: {timeframe}\t📈 Velas: {velas}")
        lines.append("")
        lines.append("")
        
        # ============================================================
        # ESTADÍSTICAS DEL MEJOR SETUP
        # ============================================================
        lines.append("-" * 70)
        lines.append("Estadísticas del mejor setup encontrado")
        lines.append("-" * 70)
        lines.append(f"Rendimiento total\t= {total_return:+.2f} % \t| Total de operaciones = {n_trades}")
        lines.append("")
        lines.append(f"Profit Factor final \t= {profit_factor:.2f}")
        lines.append(f"Win Rate total\t\t= {winrate:.2f} %")
        lines.append(f"Drawdown\t\t= {max_dd:.2f} %")
        lines.append("")
        lines.append(f"Longs (total)\t= {longs_total}\t| Longs ganadores   =\t{longs_win}\t| Rendimiento longs  = {longs_ret:+.2f} %")
        lines.append(f"Shorts (total)\t= {shorts_total}\t| Shorts ganadores  = \t{shorts_win}\t| Rendimiento shorts = {shorts_ret:+.2f} %")
        lines.append("")
        lines.append("")
        lines.append("📊 Estadísticas de Validación:")
        lines.append(f"Score final= {best_score:.4f}")
        lines.append(f"Estabilidad= {estabilidad * 100:.1f}%")
        lines.append("")
        lines.append("")
        lines.append("-" * 80)
        lines.append("📋 RESUMEN POR FASE")
        lines.append("-" * 80)
        lines.append("")
        
        # ============================================================
        # RESUMEN POR FASE (igual que antes)
        # ============================================================
        for fase in self.historial:
            fase_num = fase.get('fase')
            nombre = fase.get('nombre', f'Fase {fase_num}')
            best_score_fase = fase.get('best_score', 0)
            num_params = len(fase.get('best_params', {}))
            
            if fase_num == 1:
                cfg = self.config_fases.get('fase_1', {})
                modo = cfg.get('modo', 'paralelo')
                multi_run = cfg.get('multi_run', False)
                runs = cfg.get('runs', 1)
                trials = cfg.get('trials', 2000)
                modo_str = "Paralelo" if modo == "paralelo" else "Serie"
                multi_str = "ON" if multi_run else "OFF"
                lines.append(f"📌 {nombre} (Fase {fase_num}) - Modo {modo_str} - MultiRun {multi_str} - Corridas={runs} - Trials={trials}")
            elif fase_num == 2:
                cfg = self.config_fases.get('fase_2', {})
                modo = cfg.get('modo', 'serie')
                multi_run = cfg.get('multi_run', False)
                runs = cfg.get('runs', 1)
                trials = cfg.get('trials', 1500)
                modo_str = "Paralelo" if modo == "paralelo" else "Serie"
                multi_str = "ON" if multi_run else "OFF"
                lines.append(f"📌 {nombre} (Fase {fase_num}) - Modo {modo_str} - MultiRun {multi_str} - Corridas={runs} - Trials={trials}")
            else:
                cfg = self.config_fases.get('fase_3', {})
                modo = cfg.get('modo', 'serie')
                multi_run = cfg.get('multi_run', True)
                corridas = cfg.get('corridas', 5)
                trials_por_corrida = cfg.get('trials_por_corrida', 800)
                modo_str = "Paralelo" if modo == "paralelo" else "Serie"
                multi_str = "ON" if multi_run else "OFF"
                lines.append(f"📌 {nombre} (Fase {fase_num}) - Modo {modo_str} - MultiRun {multi_str} - Corridas={corridas} - Trials={trials_por_corrida}")
            
            lines.append(f"   • Mejor score: {best_score_fase:.4f}")
            lines.append(f"   • Parámetros: {num_params} ajustados")
            if 'estabilidad' in fase:
                lines.append(f"   • Estabilidad: {fase['estabilidad'] * 100:.1f}%")
            lines.append("")
        
        # ============================================================
        # MÉTRICAS POR FASE (igual que antes)
        # ============================================================
        lines.append("Métricas de Optimización Por Fase (Métrica - On-Off - Peso)")
        lines.append("+---------------------------------------------------------+")
        lines.append("\t\t\tFase 1\tFase 2\tFase 3")
        
        m1 = self.config_metricas.get('fase_1', {})
        m2 = self.config_metricas.get('fase_2', {})
        m3 = self.config_metricas.get('fase_3', {})
        
        lines.append(f"Profit Factor\t\t{m1.get('peso_pf', 60):.0f}%\t{m2.get('peso_pf', 40):.0f}%\t{m3.get('peso_pf', 35):.0f}%")
        lines.append(f"Winrate\t\t\t{m1.get('peso_winrate', 40):.0f}%\t{m2.get('peso_winrate', 30):.0f}%\t{m3.get('peso_winrate', 30):.0f}%")
        lines.append(f"Drawdown\t\t{m1.get('peso_drawdown', 0):.0f}%\t{m2.get('peso_drawdown', 20):.0f}%\t{m3.get('peso_drawdown', 35):.0f}%")
        lines.append(f"N° trades\t\t{m1.get('peso_n_trades', 0):.0f}%\t{m2.get('peso_n_trades', 10):.0f}%\t{m3.get('peso_n_trades', 0):.0f}%")
        lines.append(f"Min Trades\t\t{m1.get('min_trades', 15)}\t{m2.get('min_trades', 20)}\t{m3.get('min_trades', 30)}")
        lines.append("")
        lines.append("")
        
        # ============================================================
        # DIRECCIÓN DE TRADES (igual que antes)
        # ============================================================
        lines.append("-" * 80)
        lines.append("⚙️ DIRECCIÓN DE TRADES")
        lines.append("-" * 80)
        lines.append("")
        lines.append("| Dirección de Trades    |")
        lines.append("+------------------+-----+")
        lines.append(f"| Habilitar Longs  | {'ON' if self.features.get('enable_long_trades', True) else 'OFF':<3} |")
        lines.append(f"| Habilitar Shorts | {'ON' if self.features.get('enable_short_trades', True) else 'OFF':<3} |")
        lines.append("+------------------+-----+")
        lines.append("")
        
        # ============================================================
        # PRESET INICIAL vs FINAL (igual que antes)
        # ============================================================
        lines.append("=" * 100)
        lines.append("PRESET INICIAL vs FINAL")
        lines.append("=" * 100)
        lines.append("")
        
        # Medias móviles (rangos iniciales)
        lines.append("Medias móviles")
        lines.append("+------------------+-----+")
        lines.append(f"| EMA\t\t   | {'ON' if 'EMA' in self.config_base.get('tipos_ma', []) else 'OFF':<3} |")
        lines.append(f"| SMA\t\t   | {'ON' if 'SMA' in self.config_base.get('tipos_ma', []) else 'OFF':<3} |")
        lines.append(f"| WMA\t\t   | {'ON' if 'WMA' in self.config_base.get('tipos_ma', []) else 'OFF':<3} |")
        lines.append(f"| HMA\t\t   | {'ON' if 'HMA' in self.config_base.get('tipos_ma', []) else 'OFF':<3} |")
        lines.append(f"| DEMA\t\t   | {'ON' if 'DEMA' in self.config_base.get('tipos_ma', []) else 'OFF':<3} |")
        lines.append(f"| MA1 -min\t   | {self.config_base.get('ma1_min', 'N/A'):<3} |")
        lines.append(f"| MA1 -max\t   | {self.config_base.get('ma1_max', 'N/A'):<3} |")
        lines.append(f"| MA2 -min\t   | {self.config_base.get('ma2_min', 'N/A'):<3} |")
        lines.append(f"| MA2 -max\t   | {self.config_base.get('ma2_max', 'N/A'):<3} |")
        lines.append("+------------------+-----+")
        lines.append("")
        
        # Medias móviles seleccionadas
        lines.append("Medias móviles Seleccionadas")
        lines.append("+------------------+-----+")
        lines.append(f"| MA1 -tipo\t   | {best_params.get('ma1_type', 'N/A'):<3} |")
        lines.append(f"| MA1 -longitud\t   | {best_params.get('ma1_length', 'N/A'):<3} |")
        lines.append(f"| MA2 -tipo\t   | {best_params.get('ma2_type', 'N/A'):<3} |")
        lines.append(f"| MA2 -longitud\t   | {best_params.get('ma2_length', 'N/A'):<3} |")
        lines.append("+------------------+-----+")
        lines.append("")
        
        # RSI
        usar_rsi_long = best_params.get('usar_rsi_long', False)
        usar_rsi_short = best_params.get('usar_rsi_short', False)
        rsi_length = best_params.get('rsi_length', 'N/A')
        rsi_min = best_params.get('rsi_min', 'N/A')
        rsi_max = best_params.get('rsi_max', 'N/A')
        
        lines.append("| Tendencia (RSI)                                                |")
        lines.append("+-----------------+------------------------------+---------------+")
        lines.append(f"| RSI Long        | AUTO                         | Optuna: {usar_rsi_long:<12} |")
        lines.append(f"| RSI Short       | AUTO                         | Optuna: {usar_rsi_short:<12} |")
        rsi_range = self.config_base.get('rsi_length_range', (8, 18))
        lines.append(f"| RSI Length      | AUTO (min: {rsi_range[0]} | max: {rsi_range[1]})      | {rsi_length:<13} |")
        rsi_min_range = self.config_base.get('rsi_min_range', (55, 65))
        lines.append(f"| RSI min (Long)  | AUTO (min: {rsi_min_range[0]} | max: {rsi_min_range[1]}) | {rsi_min if usar_rsi_long else 'N/A':<13} |")
        rsi_max_range = self.config_base.get('rsi_max_range', (35, 45))
        lines.append(f"| RSI max (Short) | AUTO (min: {rsi_max_range[0]} | max: {rsi_max_range[1]}) | {rsi_max if usar_rsi_short else 'N/A':<13} |")
        lines.append("+-----------------+------------------------------+---------------+")
        lines.append("")
        
        # ADX
        usar_adx = best_params.get('usar_adx', False)
        adx_length = best_params.get('adx_length', 'N/A')
        adx_threshold = best_params.get('adx_threshold', 'N/A')
        
        lines.append("| ADX                                                 |")
        lines.append("+------------+------------------------------+---------+")
        lines.append(f"| ADX        | {'ON' if usar_adx else 'OFF':<28} | {'Fijo ON' if usar_adx else 'Fijo OFF':<7} |")
        adx_len_range = self.config_base.get('adx_length_range', (8, 18))
        lines.append(f"| ADX Length | AUTO (min: {adx_len_range[0]} | max: {adx_len_range[1]})      | {adx_length:<7} |")
        adx_thr_range = self.config_base.get('adx_thr_range', (15, 25))
        lines.append(f"| ADX Umbral | AUTO (min: {adx_thr_range[0]} | max: {adx_thr_range[1]})     | {adx_threshold:<7} |")
        lines.append("+------------+------------------------------+---------+")
        lines.append("")
        
        # Condiciones de Precio
        usar_high = best_params.get('usar_high', True)
        usar_low = best_params.get('usar_low', True)
        lookback = best_params.get('lookback', 'N/A')
        validation_window = best_params.get('validation_window', 'N/A')
        
        lines.append("| Condiciones de Precio                                  |")
        lines.append("+--------------------+-------------------------+---------+")
        lines.append(f"| High Condition     | {'ON' if usar_high else 'OFF':<23} | {'Fijo ON' if usar_high else 'Fijo OFF':<7} |")
        lines.append(f"| Low Condition      | {'ON' if usar_low else 'OFF':<23} | {'Fijo ON' if usar_low else 'Fijo OFF':<7} |")
        lb_range = self.config_base.get('lookback_range', (2, 10))
        lines.append(f"| Lookback           | AUTO (min: {lb_range[0]} | max: {lb_range[1]})      | {lookback:<7} |")
        val_win_auto = "ON" if self.features.get('use_validation_window', True) else "OFF"
        val_win_fijo = "Fijo ON" if self.features.get('use_validation_window', True) else "Fijo OFF"
        lines.append(f"| Validation Window  | {val_win_auto:<23} | {val_win_fijo:<7} |")
        vw_range = self.config_base.get('valwin_range', (5, 15))
        lines.append(f"| Range (Val Window) | AUTO (min: {vw_range[0]} | max: {vw_range[1]})      | {validation_window:<7} |")
        lines.append("+--------------------+-------------------------+---------+")
        lines.append("")
        
        # HTF Filter
        usar_htf = best_params.get('usar_htf', False)
        lines.append("| HTF Filter                                       |")
        lines.append("+------------+--------------------------+----------+")
        lines.append(f"| HTF Filter | {'ON' if usar_htf else 'OFF':<24} | {'Fijo ON' if usar_htf else 'Fijo OFF':<8} |")
        lines.append("| Timeframe  | N/A                      | N/A      |")
        lines.append("| MA Type    | N/A                      | N/A      |")
        htf_range = self.config_base.get('htf_length_range', (10, 60))
        lines.append(f"| HTF Length | AUTO (min: {htf_range[0]} | max: {htf_range[1]})      | {best_params.get('htf_length', 'N/A'):<8} |")
        lines.append("+------------+--------------------------+----------+")
        lines.append("")
        
        # Gestión de Riesgo
        usar_sl = best_params.get('usar_sl', False)
        usar_be = best_params.get('usar_be', False)
        usar_tp_long = best_params.get('usar_tp_long', False)
        usar_tp_short = best_params.get('usar_tp_short', False)
        sl_pct = best_params.get('stop_loss_pct', 'N/A')
        tp_long_pct = best_params.get('tp_long_pct', 'N/A')
        tp_short_pct = best_params.get('tp_short_pct', 'N/A')
        
        lines.append("| Gestión de Riesgo                                                   |")
        lines.append("+-------------------+----------------------------+--------------------+")
        lines.append(f"| Stop Loss         | {'ON' if usar_sl else 'OFF':<26} | {'Fijo ON' if usar_sl else 'Fijo OFF':<18} |")
        sl_range = self.config_base.get('sl_range', (0.3, 2.0))
        lines.append(f"| SL %              | AUTO (min: {sl_range[0]} | max: {sl_range[1]})      | {sl_pct:<18} |")
        lines.append(f"| Break Even        | {'ON' if usar_be else 'OFF':<26} | {'Fijo ON' if usar_be else 'Fijo OFF':<18} |")
        be_range = self.config_base.get('be_range', (1, 10))
        lines.append(f"| Velas para BE     | AUTO (min: {be_range[0]} | max: {be_range[1]})       | {best_params.get('velas_para_be', 'N/A'):<18} |")
        lines.append(f"| Take Profit Long  | {'AUTO' if usar_tp_long else 'OFF':<26} | {'Optuna: True' if usar_tp_long else 'Fijo OFF':<18} |")
        tp_long_range = self.config_base.get('tp_long_range', (0.5, 4.0))
        lines.append(f"| TP long %         | AUTO (min: {tp_long_range[0]} | max: {tp_long_range[1]}) | {tp_long_pct:<18} |")
        lines.append(f"| Take Profit Short | {'AUTO' if usar_tp_short else 'OFF':<26} | {'Optuna: True' if usar_tp_short else 'Fijo OFF':<18} |")
        tp_short_range = self.config_base.get('tp_short_range', (0.5, 4.0))
        lines.append(f"| TP short %        | AUTO (min: {tp_short_range[0]} | max: {tp_short_range[1]}) | {tp_short_pct:<18} |")
        lines.append("+-------------------+----------------------------+--------------------+")
        lines.append("")
        
        # Gestión Operaciones
        usar_cooldown = best_params.get('usar_cooldown', False)
        usar_reentry = best_params.get('usar_reentry', False)
        usar_post_re = best_params.get('usar_post_re', False)
        
        lines.append("| Gestión Operaciones                                         |")
        lines.append("+----------------------+---------------------------+----------+")
        lines.append(f"| Cooldown             | {'ON' if usar_cooldown else 'OFF':<25} | {'Fijo ON' if usar_cooldown else 'Fijo OFF':<8} |")
        mls_range = self.config_base.get('mls_range', (1, 4))
        lines.append(f"| Max Losing streak    | AUTO (min: {mls_range[0]} | max: {mls_range[1]})      | {best_params.get('max_losing_streak', 'N/A'):<8} |")
        cool_range = self.config_base.get('cool_range', (10, 100))
        lines.append(f"| Cooldown bars        | AUTO (min: {cool_range[0]} | max: {cool_range[1]})     | {best_params.get('cooldown_bars', 'N/A'):<8} |")
        lines.append(f"| Reentry              | {'ON' if usar_reentry else 'OFF':<25} | {'Fijo ON' if usar_reentry else 'Fijo OFF':<8} |")
        re_range = self.config_base.get('re_range', (1, 4))
        lines.append(f"| Max reentries        | AUTO (min: {re_range[0]} | max: {re_range[1]})      | {best_params.get('max_reentries_allowed', 'N/A'):<8} |")
        lines.append(f"| Post Crossover Entry | {'ON' if usar_post_re else 'OFF':<25} | {'Fijo ON' if usar_post_re else 'Fijo OFF':<8} |")
        postre_range = self.config_base.get('postre_range', (0, 3))
        lines.append(f"| Max post reentries   | AUTO (min: {postre_range[0]} | max: {postre_range[1]})      | {best_params.get('max_post_reentries', 'N/A'):<8} |")
        lines.append("+----------------------+---------------------------+----------+")
        lines.append("")
        
        # ============================================================
        # ESPACIO DE BÚSQUEDA
        # ============================================================
        lines.append("=" * 100)
        lines.append("\t\t\tESPACIO DE BÚSQUEDA")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"Dimensiones totales \t= N/A")
        lines.append(f"Complejidad estimada\t= N/A")
        lines.append(f"Tamaño estimado\t\t= N/A")
        lines.append(f"Trials recomendados\t= N/A")
        lines.append(f"Trials usados\t\t= {self.config_fases.get('fase_1', {}).get('trials', 'N/A')}")
        lines.append("")
        lines.append("")
        
        # ============================================================
        # TABLA DE OVERFITTING (IS/OOS)
        # ============================================================
        fase3 = None
        for fase in self.historial:
            if fase.get('fase') == 3:
                fase3 = fase
                break
        
        if fase3:
            pf_train = fase3.get('pf_train', 0)
            pf_test = fase3.get('pf_test', 0)
            pf_final_val = profit_factor if trades_final is not None else resultado_final.get('best_score', 0)
            winrate_train = fase3.get('winrate_train', 0)
            winrate_test = fase3.get('winrate_test', 0)
            winrate_final = winrate if trades_final is not None else 0
            drawdown_train = fase3.get('drawdown_train', 0)
            drawdown_test = fase3.get('drawdown_test', 0)
            drawdown_final = max_dd
            trades_train = fase3.get('trades_train', 0)
            trades_test = fase3.get('trades_test', 0)
            trades_final_val = n_trades
            
            lines.append("=" * 100)
            lines.append("  📊 VALIDACIÓN IS/OOS - COMPARATIVA TRAIN vs TEST vs FINAL")
            lines.append("=" * 100)
            lines.append("")
            lines.append(f"{'Métrica':<20} {'TRAIN (70%)':<18} {'TEST (30%)':<18} {'FINAL (100%)':<18}")
            lines.append("-" * 80)
            lines.append(f"{'Profit Factor':<20} {pf_train:<18.2f} {pf_test:<18.2f} {pf_final_val:<18.2f}")
            lines.append(f"{'Win Rate':<20} {winrate_train:<17.1f}% {winrate_test:<17.1f}% {winrate_final:<17.1f}%")
            lines.append(f"{'Drawdown Máx':<20} {drawdown_train:<17.2f}% {drawdown_test:<17.2f}% {drawdown_final:<17.2f}%")
            lines.append(f"{'N° Trades':<20} {trades_train:<18} {trades_test:<18} {trades_final_val:<18}")
            lines.append("-" * 80)
            
            if pf_train > 0:
                degradacion_pf = (1 - pf_test / pf_train) * 100
                if degradacion_pf < 20:
                    lines.append("\n✅ ROBUSTO: La degradación del Profit Factor es aceptable (<20%).")
                elif degradacion_pf < 40:
                    lines.append("\n⚠️ SOBREAJUSTE MODERADO: La degradación es significativa (20-40%).")
                else:
                    lines.append("\n❌ SOBREAJUSTE SEVERO: La estrategia no generaliza (>40% degradación).")
            lines.append("=" * 100)
            lines.append("")
        
        # ============================================================
        # MENSAJE FINAL
        # ============================================================
        lines.append("")
        lines.append("=" * 80)
        lines.append("🔧 Estrategia lista para usar en el optimizador manual")
        lines.append("   (Puedes cargar estos parámetros en el GUI)")
        lines.append("=" * 80)
        
        # Guardar archivo
        reporte = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(reporte)
        
        print(f"\n📄 Reporte guardado en:\n  {filepath}")

    
    
    def _guardar_seed(self, best_params, resultado_final):
        """
        Guarda un archivo JSON (seed) con toda la configuración y resultados.
        """
        from datetime import datetime
        import os
        import json
        
        # Crear carpeta de reportes si no existe
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reportes")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar timestamp y nombre de archivo
        timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        filename = f"{timestamp} Seed_AutoOptim_{safe_symbol}_{self.timeframe}.json"
        filepath = os.path.join(output_dir, filename)
        
        # ============================================================
        # IMPORTANTE: Guardar los valores del GUI si existen
        # ============================================================
        gui_values = self.gui_values #gui_values = getattr(self, 'gui_values', {})
        
        # Construir métricas para el seed
        metrics = {
            "profit_factor": resultado_final.get('best_score', 0),
            "estabilidad": resultado_final.get('estabilidad', 0),
            "trades_promedio": None,
            "corridas_validacion": len(resultado_final.get('resultados_individuales', []))
        }
        
        # Construir estructura similar a la seed manual
        seed = {
            "version": "19",
            "timestamp": timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "tipo_optimizacion": "Automatica (Multi-Fase)",
            
            # ============================================================
            # NUEVO: Guardar TODOS los valores del GUI
            # ============================================================
            "gui_values": gui_values,
            
            "configuracion_automatica": {
                "fase_1_trials": self.config_fases.get("fase_1", {}).get("trials", 2000),
                "fase_1_modo": self.config_fases.get("fase_1", {}).get("modo", "paralelo"),
                "fase_2_trials": self.config_fases.get("fase_2", {}).get("trials", 1500),
                "fase_2_modo": self.config_fases.get("fase_2", {}).get("modo", "serie"),
                "fase_3_corridas": self.config_fases.get("fase_3", {}).get("corridas", 5),
                "fase_3_trials_por_corrida": self.config_fases.get("fase_3", {}).get("trials_por_corrida", 800),
                "min_trades_fase_1": self.config_metricas.get("fase_1", {}).get("min_trades", 15),
                "min_trades_fase_2": self.config_metricas.get("fase_2", {}).get("min_trades", 20),
                "min_trades_fase_3": self.config_metricas.get("fase_3", {}).get("min_trades", 30),
                "convergencia_activada": self.config_convergencia.get("activar", True)
            },
            
            "best_params": best_params,
            "metrics": metrics,
            "historial_fases": self.historial,
            
            "output_files": {
                "reporte_txt": None,
                "seed_json": filepath
            }
        }
        
        # Guardar archivo
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(seed, f, indent=4, ensure_ascii=False)
        
        print(f"\n📦 Seed (JSON) guardado en:\n  {filepath}")


        

    def ejecutar(self):
        """
        Ejecuta el flujo completo de optimización automática.
        
        Returns:
            dict: Resultados finales con mejor score, parámetros y estadísticas
        """
        self._log("\n" + "🚀"*30)
        self._log(" OPTIMIZADOR AUTOMÁTICO MULTI-FASE")
        self._log(f" Activo: {self.symbol} | Timeframe: {self.timeframe}")
        self._log("🚀"*30)
        
        # Fase 1: Exploración
        params_fase1 = self._fase_exploracion_rapida()
        
        # Fase 2: Refinamiento
        params_fase2 = self._fase_refinamiento(params_fase1)
        
        # Fase 3: Validación
        params_fase3 = self._fase_validacion(params_fase2)
        
        # Resultado final
        resultado_final = self.historial[-1]
        
        # ============================================================
        # OBTENER TRADES Y EQUITY DEL BACKTEST FINAL
        # ============================================================
        from optimizador_main import run_backtest
        from config import CONSTANTS
        
        # Limpiar features (quitar "auto")
        cleaned_features = {k: v for k, v in self.features.items() if v != "auto"}
        
        # ============================================================
        # MAPEAR PARÁMETROS DE OPTUNA A LOS NOMBRES QUE ESPERA run_backtest
        # ============================================================
        MAPEO_AUTO_BACKTEST = {
            "usar_rsi_long": "use_rsi_long",
            "usar_rsi_short": "use_rsi_short",
            "usar_adx": "use_adx_filter",
            "usar_high": "enable_high_condition",
            "usar_low": "enable_low_condition",
            "usar_htf": "use_htf_filter",
            "usar_sl": "use_stop_loss",
            "usar_be": "activar_stop_be",
            "usar_tp_long": "use_take_profit_long",
            "usar_tp_short": "use_take_profit_short",
            "usar_cooldown": "enable_cooldown",
            "usar_reentry": "enable_reentry",
            "usar_post_re": "enable_post_crossover_entry"
        }
        
        # Traducir parámetros de Optuna a nombres correctos
        cleaned_params = {}
        for k, v in params_fase3.items():
            if k in MAPEO_AUTO_BACKTEST:
                cleaned_params[MAPEO_AUTO_BACKTEST[k]] = v
            else:
                cleaned_params[k] = v
        
        # Ejecutar backtest final con los mejores parámetros traducidos
        pf_final, equity_curve_final, trades_final = run_backtest(
            self.df,  # Datos completos
            **cleaned_params,
            **cleaned_features,
            **CONSTANTS
        )
        
        self._log("\n" + "="*60)
        self._log("✅ OPTIMIZACIÓN AUTOMÁTICA COMPLETADA")
        self._log("="*60)
        self._log(f"\n📊 Mejor score final: {resultado_final['best_score']:.4f}")
        self._log(f"🎯 Estabilidad: {resultado_final.get('estabilidad', 'N/A')}")
        self._log(f"📈 Profit Factor final (backtest): {pf_final:.4f}")
        self._log(f"💰 Trades totales: {len(trades_final)}")
        
        # GENERAR REPORTE TXT (pasando trades_final y equity_curve_final)
        self._generar_reporte_txt(
            best_params=params_fase3,
            resultado_final=resultado_final,
            trades_final=trades_final,
            equity_curve_final=equity_curve_final,
            pf_final=pf_final
        )
        
        # GUARDAR SEED JSON
        self._guardar_seed(params_fase3, resultado_final)
        
        return resultado_final
    



    def reporte(self):
        """
        Genera reporte resumido del proceso de optimización.
        """
        print("\n" + "="*70)
        print("  📈 REPORTE DE OPTIMIZACIÓN AUTOMÁTICA")
        print("="*70)
        
        for fase in self.historial:
            print(f"\n📌 {fase['nombre']} (Fase {fase['fase']})")
            print(f"   • Mejor score: {fase['best_score']:.4f}")
            print(f"   • Parámetros: {len(fase.get('best_params', {}))} ajustados")
            if 'estabilidad' in fase:
                print(f"   • Estabilidad: {fase['estabilidad']:.1%}")
        
        print("\n" + "="*70)



def ejecutar_optimizacion_automatica(df, config_base, features, symbol, timeframe, 
    config_fases=None, config_convergencia=None,
    config_metricas=None, gui_values=None, 
    df_train=None, df_test=None, constants=None, verbose=True):
    """
    Función de alto nivel para ejecutar la optimización automática.
    
    Args:
        df: DataFrame con datos históricos completos
        df_train: DataFrame para entrenamiento (opcional)
        df_test: DataFrame para validación (opcional)
        constants: Constantes de configuración (comisión, capital, etc.)
        ... otros args ...
    """
    optimizador = OptimizadorAutomatico(
        df=df,
        df_train=df_train,
        df_test=df_test,
        constants=constants,
        config_base=config_base,
        features=features,
        symbol=symbol,
        timeframe=timeframe,
        config_fases=config_fases,
        config_convergencia=config_convergencia,
        config_metricas=config_metricas,
        gui_values=gui_values,
        verbose=verbose
    )
    
    resultado = optimizador.ejecutar()
    optimizador.reporte()
    
    return resultado